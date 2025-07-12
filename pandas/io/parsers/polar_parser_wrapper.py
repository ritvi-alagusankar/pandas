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
        if self.index_col is not None and self.index_col is not False:
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

        # Direct parameter mappings
        pandas_map = {
            "sep": "separator",
            "delimiter": "separator", 
            "names": "new_columns",
            "quotechar": "quote_char",
            "comment": "comment_prefix",
            "storage_options": "storage_options",
            "encoding": "encoding",
            "low_memory": "low_memory",
        }

        # Apply direct mappings
        for pd_key, pl_key in pandas_map.items():
            if pd_key in opts:
                val = opts[pd_key]
                if val is not None:
                    polars_kwargs[pl_key] = val

                # Handle header parameter
        header = opts["header"]
        skiprows = opts.get("skiprows", 0)
        if header in ("infer", 0):
            polars_kwargs["has_header"] = True
        elif header is None:
            polars_kwargs["has_header"] = False
        elif isinstance(header, list) or (isinstance(header, int) and header != 0):
            if isinstance(header, int):
                skiprows = header
                polars_kwargs["has_header"] = True
            elif isinstance(header, list) and len(header) == 1 and isinstance(header[0], int):
                skiprows = header[0]
                polars_kwargs["has_header"] = True
            else:
                raise NotImplementedError(
                    "Polars does not support multiple header rows"
                )
            
        # Handle skip rows
        if isinstance(skiprows, int):
            polars_kwargs["skip_rows"] = skiprows
        elif isinstance(skiprows, (list, tuple)):
            if len(skiprows) == 0:
                polars_kwargs["skip_rows"] = 0
            elif len(skiprows) == 1 and isinstance(skiprows[0], int):
                polars_kwargs["skip_rows"] = skiprows[0]
            else:
                raise NotImplementedError(
                    "Polars does not support skipping multiple rows by list or tuple of integers."
                )
        elif callable(skiprows):
            raise NotImplementedError("Polars does not support callable skiprows argument.")


        if "usecols" in opts:
            usecols = opts["usecols"]
            if callable(usecols):
                raise NotImplementedError("Polars does not support callable usecols argument")
            else:
                polars_kwargs["columns"] = usecols  

        if "lineterminator" in opts:
            lineterminator = opts["lineterminator"]
            if lineterminator is not None:
                polars_kwargs["eol_char"] = lineterminator

        if "decimal" in opts:
            decimal = opts["decimal"]
            if decimal == ",":
                polars_kwargs["decimal_comma"] = True
            elif decimal == ".":
                polars_kwargs["decimal_comma"] = False
            else:
                raise NotImplementedError(
                    f"Polars only supports '.' or ',' as decimal separator, got '{decimal}'"
                )

        if hasattr(self, 'parse_dates') and self.parse_dates is not None:
            if isinstance(self.parse_dates, bool):
                polars_kwargs["try_parse_dates"] = self.parse_dates
            else:
                raise NotImplementedError(
                    "Polars does not support date parsing with `parse_dates` of specific columns. Use" \
                    "only boolean `parse_dates` to enable date parsing for all columns."
                )

        on_bad_lines = opts.get("on_bad_lines", "error")
        if on_bad_lines == "error":
            polars_kwargs["raise_if_empty"] = True
            polars_kwargs["ignore_errors"] = False
        elif on_bad_lines in {"warn", "skip"}:
            polars_kwargs["raise_if_empty"] = False
            polars_kwargs["ignore_errors"] = True


        # # Warn about unsupported parameters that are being used
        # for param in unsupported:
        #     if param in opts and opts[param] is not None:    
        #         warnings.warn(
        #             f"Parameter '{param}' is not supported in Polars and will be ignored.",
        #             UserWarning
        #         )

        return polars_kwargs
