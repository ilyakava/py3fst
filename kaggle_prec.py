import tensorflow as tf

from tensorflow.python.training import distribute as distribute_lib
# in HEAD:
# from tensorflow.python.training import distribution_strategy_context
# https://github.com/tensorflow/tensorflow/commit/77fabbeabb5b9061d8c606050c1ea79aec990c03
def _aggregate_across_towers(metrics_collections, metric_value_fn, *args):
    """Aggregate metric value across towers."""
    def fn(distribution, *a):
        """Call `metric_value_fn` in the correct control flow context."""
        if hasattr(distribution, '_outer_control_flow_context'):
            # If there was an outer context captured before this method was called,
            # then we enter that context to create the metric value op. If the
            # caputred context is `None`, ops.control_dependencies(None) gives the
            # desired behavior. Else we use `Enter` and `Exit` to enter and exit the
            # captured context.
            # This special handling is needed because sometimes the metric is created
            # inside a while_loop (and perhaps a TPU rewrite context). But we don't
            # want the value op to be evaluated every step or on the TPU. So we
            # create it outside so that it can be evaluated at the end on the host,
            # once the update ops have been evaluted.

            # pylint: disable=protected-access
            if distribution._outer_control_flow_context is None:
                with ops.control_dependencies(None):
                    metric_value = metric_value_fn(distribution, *a)
            else:
                distribution._outer_control_flow_context.Enter()
                metric_value = metric_value_fn(distribution, *a)
                distribution._outer_control_flow_context.Exit()
                # pylint: enable=protected-access
        else:
            metric_value = metric_value_fn(distribution, *a)
        if metrics_collections:
            ops.add_to_collections(metrics_collections, metric_value)
        return metric_value

    return distribute_lib.get_tower_context().merge_call(fn, *args)

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
def kaggle_prec(labels, predictions, weights=None, metrics_collections=None, updates_collections=None, name=None):
    # https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/python/ops/metrics_impl.py#L1936
    true_p, true_positives_update_op = tf.metrics.true_positives(
        labels,
        predictions,
        weights=None,
        metrics_collections=None,
        updates_collections=None,
        name=None)
    false_p, false_positives_update_op = tf.metrics.false_positives(
        labels,
        predictions,
        weights=None,
        metrics_collections=None,
        updates_collections=None,
        name=None)
    false_n, false_negatives_update_op = tf.metrics.false_negatives(
        labels,
        predictions,
        weights=None,
        metrics_collections=None,
        updates_collections=None,
        name=None)

    def compute_precision(tp, fp, fn, name):
        return array_ops.where(
            math_ops.greater(tp + fp + fn, 0), math_ops.div(tp, tp + fp + fn), 0, name)

    def once_across_towers(_, true_p, false_p, false_n):
        return compute_precision(true_p, false_p, false_n, 'value')

    p = _aggregate_across_towers(metrics_collections, once_across_towers,
                                 true_p, false_p, false_n)

    update_op = compute_precision(true_positives_update_op,
                                  false_positives_update_op, false_negatives_update_op, 'update_op')

    return p, update_op
