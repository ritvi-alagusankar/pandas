from pandas.errors import ParserError
from pandas.compat._optional import import_optional_dependency
from pandas.io.parsers.base_parser import ParserBase

from pandas.core.dtypes.common import pandas_dtype
from pandas.core.dtypes.inference import is_integer

class PolarsParserWrapper(ParserBase):
    """
    CSV parser using Polars as the backend engine.
    """

    def __init__(self, src, **kwds):
        super().__init__(kwds)
        self.kwds = kwds
        self.src = src 

    def read(self, nrows=None):
        """
        Read the CSV using Polars' lazy API.
        """
        try:
            df = self._read_csv_with_polars(nrows)
        except Exception as err:
            raise ParserError(f"Polars CSV parser error: {err}") from err
        return df

    def _read_csv_with_polars(self, nrows):
        pl = import_optional_dependency("polars")
        kwds = self._translate_kwargs()
        if nrows is not None:
            kwds["n_rows"] = nrows 
        lf = pl.read_csv(self.src, **kwds).lazy()
        df = lf.collect().to_pandas()
        return self._finalize_pandas_output(df)


    def _finalize_pandas_output(self, frame):
        """
        Processes data read in based on kwargs.

        Parameters
        ----------
        frame: DataFrame
            The DataFrame to process.

        Returns
        -------
        DataFrame
            The processed DataFrame.
        """
        num_cols = len(frame.columns)
        multi_index_named = True
        if self.header is None:
            if self.names is None:
                if self.header is None:
                    self.names = range(num_cols)
            if len(self.names) != num_cols:
                columns_prefix = [str(x) for x in range(num_cols - len(self.names))]
                self.names = columns_prefix + self.names
                multi_index_named = False
            frame.columns = self.names

        frame = self._do_date_conversions(frame.columns, frame)
        if self.index_col is not None:
            index_to_set = self.index_col.copy()
            for i, item in enumerate(self.index_col):
                if is_integer(item):
                    index_to_set[i] = frame.columns[item]
                elif item not in frame.columns:
                    raise ValueError(f"Index {item} invalid")

                if self.dtype is not None:
                    key, new_dtype = (
                        (item, self.dtype.get(item))
                        if self.dtype.get(item) is not None
                        else (frame.columns[item], self.dtype.get(frame.columns[item]))
                    )
                    if new_dtype is not None:
                        frame[key] = frame[key].astype(new_dtype)
                        del self.dtype[key]

            frame.set_index(index_to_set, drop=True, inplace=True)
            # Clear names if headerless and no name given
            if self.header is None and not multi_index_named:
                frame.index.names = [None] * len(frame.index.names)

        if self.dtype is not None:
            if isinstance(self.dtype, dict):
                self.dtype = {
                    k: pandas_dtype(v)
                    for k, v in self.dtype.items()
                    if k in frame.columns
                }
            else:
                self.dtype = pandas_dtype(self.dtype)
            try:
                frame = frame.astype(self.dtype)
            except TypeError as err:
                # GH#44901 reraise to keep api consistent
                raise ValueError(str(err)) from err
        return frame

    def _translate_kwargs(self):
        """
        Translate pandas read_csv kwargs to Polars-compatible kwargs.
        """
        opts = self.kwds.copy()
        polars_kwargs = {}

        mapping = {
            "sep": ("separator", lambda v: v),
            "delimiter": ("separator", lambda v: v),
            "names": ("new_columns", lambda v: v),
            "quotechar": ("quote_char", lambda v: v),
            "comment": ("comment_prefix", lambda v: v),
            "storage_options": ("storage_options", lambda v: v),
        }

        # Handling for header options
        header = opts.get("header", "infer")

        if header in ("infer", 0):
            polars_kwargs["has_header"] = True
        elif header is None:
            polars_kwargs["has_header"] = False
        elif isinstance(header, list) or (isinstance(header, int) and header != 0):
            raise NotImplementedError(
                f"Polars does not support `header={header}`. "
                "Only `header=0`, `header=None`, or `header='infer'` are supported. "
                "Multi-row headers or specifying a header row beyond the first is not supported."
            )
        
        # Handling for column selection
        if "usecols" in opts:
            usecols = opts["usecols"]
            if callable(usecols):
                raise NotImplementedError("Polars does not support callable usecols argument")
            else:
                polars_kwargs["columns"] = usecols
        
        # Handling number of rows to be skipped while parsing
        if "skiprows" in opts:
            skiprows = opts["skiprows"]
            if len(skiprows) == 0:
                polars_kwargs["skip_rows"] = 0
            elif len(skiprows) == 1 and isinstance(skiprows[0], int):
                polars_kwargs["skip_rows"] = skiprows[0]
            else:
                raise NotImplementedError(
                    "Polars does not support skipping multiple rows or callable skiprows argument"
                )

        # Handling date parsing options
        if isinstance(self.parse_dates, bool):
            polars_kwargs["try_parse_dates"] = self.parse_dates

        # Translate options to Polars kwargs
        for pd_key, (pl_key, transform) in mapping.items():
            if pd_key in opts:
                val = transform(opts[pd_key])
                if val is not None:
                    polars_kwargs[pl_key] = val

        on_bad_lines = opts.get("on_bad_lines", "error")
        polars_kwargs["raise_if_empty"] = on_bad_lines == "error"
        polars_kwargs["ignore_errors"] = on_bad_lines in {"warn", "skip"}

        return polars_kwargs
