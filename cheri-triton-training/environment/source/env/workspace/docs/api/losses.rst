Losses
======

.. module:: cherimoya.losses


_mixture_loss
-------------

.. autofunction:: _mixture_loss


Imported Losses
---------------

The following loss function is imported from ``bpnetlite`` and used
internally by :func:`_mixture_loss`:

.. function:: MNLLLoss(logps, true_counts)

   Multinomial negative log-likelihood loss. Computes the negative log
   probability of the observed counts under a multinomial distribution
   parameterized by the predicted log probabilities.

   :param logps: Predicted log probabilities, shape ``(n, length)``
   :param true_counts: Observed integer counts, shape ``(n, length)``
   :returns: Loss per example, shape ``(n,)``
