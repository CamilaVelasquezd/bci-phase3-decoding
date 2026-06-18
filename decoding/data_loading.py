from __future__ import annotations

import configparser
import os

import xarray as xr
from bci_decoding_dataset import DatasetLoader


def merge_session_attrs(
    loader: DatasetLoader, session_id: str, ds: xr.Dataset
) -> None:
    """Merge top-level session zarr attrs into ds.attrs to expose subject_id.

    Processed data attrs take precedence on key conflicts.
    """
    top_level_attrs = dict(loader.combined_zarr[session_id].attrs)
    ds.attrs.update({**top_level_attrs, **ds.attrs})


def load_session(
    dataset_id: str, session_index: int = 0
) -> tuple[DatasetLoader, xr.Dataset, str]:
    """Load a single session from S3 for a given dataset.

    Reads AWS credentials from ``~/.aws/credentials`` using the profile
    specified by the ``AWS_PROFILE`` environment variable (default ``cv-pc``).

    Parameters
    ----------
    dataset_id : str
        Dataset identifier as stored in the session catalog,
        e.g. ``'DANDI_00070'``, ``'DANDI_000688'``, ``'Zenodo_3854034'``.
    session_index : int
        Zero-based index into the list of sessions for this dataset. Default 0.

    Returns
    -------
    loader : DatasetLoader
        Connected DatasetLoader instance; reuse it for further calls to avoid
        re-reading credentials and rebuilding the S3 connection.
    ds : xr.Dataset
        Loaded session dataset (lazy — slice before calling ``.values``).
    session_id : str
        Session identifier used to load ``ds``.

    Raises
    ------
    IndexError
        If ``session_index`` is out of range for ``dataset_id``.
    KeyError
        If the AWS profile is not found in ``~/.aws/credentials``.
    """
    credentials_path = os.path.expanduser("~/.aws/credentials")
    config = configparser.ConfigParser()
    config.read(credentials_path)
    profile = os.environ.get("AWS_PROFILE", "cv-pc")

    loader = DatasetLoader(
        aws_store=True,
        s3_bucket="solzbacher-lab-motor-decoding-ds",
        s3_key="datasets/Combined_Motor_Datasets",
        aws_access_key_id=config[profile]["aws_access_key_id"],
        aws_secret_access_key=config[profile]["aws_secret_access_key"],
    )

    sessions = loader.filter_sessions("dataset_id", dataset_id)
    session_id = sessions[session_index]
    ds = loader.get_processed_data_from_session(session_id)
    merge_session_attrs(loader, session_id, ds)
    
    return loader, ds, session_id
