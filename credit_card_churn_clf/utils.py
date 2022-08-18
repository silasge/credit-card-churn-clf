from functools import partial, wraps
from typing import Callable, Literal

import joblib
from loguru import logger


def export_pkl(func: Callable = None, verbose: bool = False):
    if func is None:
        return partial(export_pkl, verbose=verbose)

    @wraps(func)
    def decorator(*args, export_pkl_to=None, **kwargs):
        obj = func(*args, export_pkl_to=export_pkl_to, **kwargs)
        if export_pkl_to:
            joblib.dump(obj, export_pkl_to)
        return obj

    return decorator


def add_logger(func: Callable = None, type: Literal["export", "training"] = "export"):
    if func is None:
        return partial(add_logger, type=type)

    @wraps(func)
    def decorator(*args, print_log=False, **kwargs):
        if print_log:
            if type == "export":
                logger.info(f"Criando arquivo {kwargs['export_pkl_to']}...")
                obj = func(*args, print_log=print_log, **kwargs)
                logger.info("Arquivo criado com sucesso.")
                return obj
            elif type == "training":
                logger.info(
                    f"""Treinando modelo {kwargs['model']} com {kwargs['cv']} folds 
                    e métrica {kwargs['scoring']} em {kwargs['n_iter']} iterações..."""
                )
                model_obj = func(*args, print_log=print_log, **kwargs)
                if kwargs["export_pkl_to"]:
                    logger.info(f"Modelo salvo em {kwargs['export_pkl_to']}.")
                return model_obj
        else:
            obj = func(*args, print_log=print_log, **kwargs)
            return obj

    return decorator
