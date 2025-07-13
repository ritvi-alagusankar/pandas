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
        """
        num_cols = len(frame.columns)
        multi_index_named = True
        if self.header is None:
            if self.names is None:
                self.names = list(range(num_cols))
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
            "low_memory": "low_memory",
        }

        # Apply direct mappings
        for pd_key, pl_key in pandas_map.items():
            if pd_key in opts:
                val = opts[pd_key]
                if val is not None:
                    polars_kwargs[pl_key] = val

        # Handle header and skiprows
        header = opts.get("header", "infer")
        skiprows = opts.get("skiprows", 0) or 0  # Default to 0 if skiprows is None

        num_skiprows = 0
        if isinstance(skiprows, int):
            if skiprows < 0:
                raise ValueError(
                    f"skiprows must be a non-negative integer, got {skiprows}"
                )
            num_skiprows = skiprows
        elif isinstance(skiprows, (list, tuple)):
            if len(skiprows) == 0:
                num_skiprows = 0
            elif len(skiprows) == 1 and isinstance(skiprows[0], int):
                if skiprows[0] < 0:
                    raise ValueError(
                        f"skiprows must be a non-negative integer, got {skiprows[0]}"
                    )
                num_skiprows = skiprows[0]
            else:
                raise NotImplementedError(
                    "Polars does not support skipping multiple rows by a list/tuple."
                )
        elif callable(skiprows):
            raise NotImplementedError(
                "Polars does not support callable skiprows argument."
            )
        else:
            raise TypeError(
                f"skiprows must be int, list, tuple, or callable, got {type(skiprows)}"
            )

        if header is None:
            polars_kwargs["has_header"] = False
            polars_kwargs["skip_rows"] = num_skiprows

        else:
            if header == "infer" or header == 0:
                polars_kwargs["skip_rows"] = num_skiprows
            elif isinstance(header, int):
                if header < 0:
                    raise ValueError(
                        f"header must be a non-negative integer, got {header}"
                    )
                polars_kwargs["skip_rows"] = num_skiprows + header
            elif isinstance(header, list):
                if len(header) == 1 and isinstance(header[0], int):
                    if header[0] < 0:
                        raise ValueError(
                            f"header must be a non-negative integer, got {header[0]}"
                        )
                    polars_kwargs["skip_rows"] = num_skiprows + header[0]
                else:
                    raise NotImplementedError(
                        "Polars does not support multiple header rows"
                    )
            else:
                raise TypeError(
                    f"header must be None, 'infer', int, or list of int, got {type(header)}"
                )

            polars_kwargs["has_header"] = True

        # handle encoding and encoding errors
        if "encoding" in opts and opts["encoding"] is not None:
            encoding = opts["encoding"]

            if "encoding_errors" in opts:
                encoding_errors = opts["encoding_errors"]

                if encoding_errors == "replace":
                    encoding = f"{encoding}-lossy"
                elif encoding_errors != "strict":
                    raise ValueError(
                        f"Invalid value for encoding_errors: {encoding_errors}. "
                        "The polars engine only supports 'strict' or 'replace'."
                    )

            polars_kwargs["encoding"] = encoding

        if "usecols" in opts:
            usecols = opts["usecols"]
            if callable(usecols):
                raise NotImplementedError(
                    "Polars does not support callable usecols argument"
                )
            else:
                polars_kwargs["columns"] = usecols

        if "lineterminator" in opts:
            lineterminator = opts["lineterminator"]
            if isinstance(lineterminator, str):
                if len(lineterminator) == 1:
                    polars_kwargs["eol_char"] = lineterminator
                else:
                    raise NotImplementedError(
                        f"Polars does not support multi-character line terminators, got '{lineterminator}'"
                    )

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

        if "parse_dates" in opts:
            parse_dates = opts["parse_dates"]
            if isinstance(parse_dates, bool):
                polars_kwargs["try_parse_dates"] = parse_dates
            else:
                raise NotImplementedError(
                    "Polars does not support date parsing with `parse_dates` of specific columns. "
                    "Use only `parse_dates=True` to enable date parsing for all columns."
                )

        if "on_bad_lines" in opts:
            on_bad_lines = opts["on_bad_lines"]

            if callable(on_bad_lines):
                raise NotImplementedError(
                    "Polars does not support callable on_bad_lines argument"
                )
            elif on_bad_lines == ParserBase.BadLineHandleMethod.ERROR:
                polars_kwargs["ignore_errors"] = False

            elif on_bad_lines in {
                ParserBase.BadLineHandleMethod.WARN,
                ParserBase.BadLineHandleMethod.SKIP,
            }:
                polars_kwargs["ignore_errors"] = True
            else:
                raise ValueError(
                    f"Unrecognized value for on_bad_lines: {on_bad_lines}. "
                )

        return polars_kwargs
