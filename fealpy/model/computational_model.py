from typing import List, Optional
import inspect
import logging

from ..logs import TqdmLoggingHandler
from ..decorator.variantmethod import Variantmethod


__all__ = ['ComputationalModel', ]
_PK = inspect._ParameterKind


class ComputationalModel:
    def __init__(self, pbar_log=False, log_level="WARNING"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.propagate = False
        self.logger.setLevel(log_level)

        if pbar_log:
            self.logger.addHandler(TqdmLoggingHandler())
        else:
            from ..logs import handler
            self.logger.addHandler(handler)

    @staticmethod
    def _get_func_params(func):
        """Make function signature"""
        sig = inspect.signature(func)
        params = []
        prev_kind = None

        for name, param in sig.parameters.items():
            kind = param.kind
            # Insert '/' for positional-only, '*' for keyword-only
            if prev_kind is not None:
                if prev_kind == _PK.POSITIONAL_ONLY and kind != _PK.POSITIONAL_ONLY:
                    params.append('/')
                if prev_kind != _PK.KEYWORD_ONLY and kind == _PK.KEYWORD_ONLY:
                    params.append('*')
            elif kind == _PK.KEYWORD_ONLY:
                params.append('*')
            # Format parameter
            if param.default is param.empty:
                if kind == _PK.VAR_POSITIONAL:
                    params.append(f'*{name}')
                elif kind == _PK.VAR_KEYWORD:
                    params.append(f'**{name}')
                else:
                    params.append(name)
            else:
                params.append(f"{name}={param.default!r}")
            prev_kind = kind
        # Add '/' if the last parameter is positional-only
        if params and list(sig.parameters.values())[-1].kind == _PK.POSITIONAL_ONLY:
            params.append('/')
        # Remove duplicate '/' or '*' if present
        result = []
        for i, p in enumerate(params):
            if i > 0 and p in ('/', '*') and params[i-1] == p:
                continue
            result.append(p)
        return '(' + ', '.join(result) + ')'

    @classmethod
    def _help_impl(cls, attr_name: str, show_docs=True, full_docs=False, show_params=True):
        attr = getattr(cls, attr_name, None)

        if isinstance(attr, Variantmethod):
            var_list = []
            default_key = attr.default_key
            func = attr.__func__

            for var in attr.virtual_table.keys():
                if var == default_key:
                    var_list.append(str(var) + '*')
                else:
                    var_list.append(str(var))

            header = f'{attr_name} [{" | ".join(var_list)}]'
        elif callable(attr):
            func = attr
            header = attr_name
        else:
            return None

        if show_params:
            header += " " + cls._get_func_params(func)

        if show_docs:
            doc = func.__doc__
            if doc is None:
                doc = 'No description'
            elif full_docs:
                doc = doc.strip()
            else:
                doc = doc.split('\n')[0].strip()

            return header + '\n    ' + doc
        else:
            return header + '\n'

    @classmethod
    def help(cls, name: Optional[str] = None, /, show_docs=True, full_docs=False, show_params=True):
        """Return a help string for the class.

        Parameters:
            name (str | None, optional): The name of the attribute to get help for.
                If None, returns help for all attributes.
            show_docs (bool, optional): Whether to include the documentation in the help string.
            full_docs (bool, optional): If True, shows the full documentation;
                otherwise, shows a brief description (the first line).
            show_params (bool, optional): Whether to include function parameters in the help string.

        Returns:
            A string containing the help information for the class or the specified attribute.
        """
        if name is None:
            attr_info: List[str] = []

            for attr_name in (s for s in dir(cls) if not s.startswith('_')):
                info = cls._help_impl(attr_name, show_docs, full_docs, show_params)
                if info is not None:
                    attr_info.append(info)

            return '\n\n'.join(attr_info)
        else:
            return cls._help_impl(name, show_docs, full_docs, show_params)
