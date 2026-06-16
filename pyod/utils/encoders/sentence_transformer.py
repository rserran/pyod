# -*- coding: utf-8 -*-
"""SentenceTransformerEncoder for EmbeddingOD."""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from . import BaseEncoder


class SentenceTransformerEncoder(BaseEncoder):
    """Encoder using sentence-transformers library.

    Wraps ``sentence_transformers.SentenceTransformer`` to produce
    text embeddings compatible with PyOD detectors.

    Parameters
    ----------
    model_name : str or SentenceTransformer instance, optional
        (default='all-MiniLM-L6-v2')
        - If str: model name (HF Hub ID) OR local filesystem path.
          Local path is detected and loaded with ``local_files_only=True``
          to prevent any network call.
        - If SentenceTransformer instance: used directly, skipping load.
          Useful for air-gapped environments or when you need custom
          model configuration not exposed via constructor params.

    device : str or None, optional (default=None)
        Device for inference ('cpu', 'cuda', etc.).
        None for auto-detection.

    normalize : bool, optional (default=False)
        L2-normalize output embeddings.

    truncate_dim : int or None, optional (default=None)
        Truncate embeddings to this dimensionality (Matryoshka).

    Examples
    --------
    >>> from pyod.utils.encoders.sentence_transformer import \\
    ...     SentenceTransformerEncoder
    >>> encoder = SentenceTransformerEncoder('all-MiniLM-L6-v2')
    >>> embeddings = encoder.encode(["hello world", "anomaly text"])
    >>> embeddings.shape
    (2, 384)

    # Local filesystem path (air-gapped)
    >>> enc = SentenceTransformerEncoder('/mnt/models/my-weights')

    # Pre-instantiated model object
    >>> my_model = SentenceTransformer('all-MiniLM-L6-v2')
    >>> enc = SentenceTransformerEncoder(my_model)
    """

    def __init__(self, model_name='all-MiniLM-L6-v2', device=None,
                 normalize=False, truncate_dim=None):
        if SentenceTransformer is None:
            raise ImportError(
                "SentenceTransformerEncoder requires 'sentence-transformers'. "
                "Install with: pip install sentence-transformers")
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self.truncate_dim = truncate_dim

    def encode(self, X, batch_size=32, show_progress=True):
        """Encode text strings to embeddings.

        Parameters
        ----------
        X : list of str
            Text strings to encode.

        batch_size : int, optional (default=32)
            Batch size for encoding.

        show_progress : bool, optional (default=True)
            Show progress bar.

        Returns
        -------
        embeddings : numpy array of shape (n_samples, n_features)
        """
        if not hasattr(self, 'model_'):
            if isinstance(self.model_name, str):
                import os
                if os.path.exists(self.model_name):
                    # Local path: load without hitting HF Hub
                    self.model_ = SentenceTransformer(
                        self.model_name,
                        device=self.device,
                        local_files_only=True,
                    )
                else:
                    # Remote/registry name: existing behavior
                    self.model_ = SentenceTransformer(
                        self.model_name, device=self.device)
            else:
                # Pre-instantiated SentenceTransformer object
                if not isinstance(self.model_name, SentenceTransformer):
                    raise TypeError(
                        "model_name must be a str or SentenceTransformer "
                        "instance, got %s" % type(self.model_name))
                self.model_ = self.model_name

        embeddings = self.model_.encode(
            X,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            truncate_dim=self.truncate_dim,
        )
        return self._validate_output(embeddings, n_samples=len(X))
