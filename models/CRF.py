import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Optimizer
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
class CRF(tf.keras.layers.Layer):
    """
        Conditional Random Field layer (tf.keras)
        `CRF` can be used as the last layer in a network (as a classifier). Input shape (features)
        must be equal to the number of classes the CRF can predict (a linear layer is recommended).
        Note: the loss and accuracy functions of networks using `CRF` must
        use the provided loss and accuracy functions (denoted as loss and viterbi_accuracy)
        as the classification of sequences are used with the layers internal weights.
        Args:
            output_dim (int): the number of labels to tag each temporal input.
        Input shape:
            nD tensor with shape `(batch_size, sentence length, num_classes)`.
        Output shape:
            nD tensor with shape: `(batch_size, sentence length, num_classes)`.
        """

    def __init__(self,
                 output_dim,
                 mode='reg',
                 supports_masking=False,
                 transitions=None,
                 **kwargs):
        self.transitions = None
        super(CRF, self).__init__(**kwargs)
        self.output_dim = int(output_dim)
        self.mode = mode
        if self.mode == 'pad':
            self.input_spec = [tf.keras.layers.InputSpec(min_ndim=3), tf.keras.layers.InputSpec(min_ndim=2)]
        elif self.mode == 'reg':
            self.input_spec = tf.keras.layers.InputSpec(min_ndim=3)
        else:
            raise ValueError
        self.supports_masking = supports_masking
        self.sequence_lengths = None

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'mode': self.mode,
            'supports_masking': self.supports_masking,
            'transitions': tf.keras.backend.eval(self.transitions)
        }
        base_config = super(CRF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        if self.mode == 'pad':
            assert len(input_shape) == 2
            assert len(input_shape[0]) == 3
            assert len(input_shape[1]) == 2
            f_shape = tf.TensorShape(input_shape[0])
            input_spec = [tf.keras.layers.InputSpec(min_ndim=3, axes={-1: f_shape[-1]}),
                          tf.keras.layers.InputSpec(min_ndim=2, axes={-1: 1}, dtype=tf.int32)]
        else:
            assert len(input_shape) == 3
            f_shape = tf.TensorShape(input_shape)
            input_spec = tf.keras.layers.InputSpec(min_ndim=3, axes={-1: f_shape[-1]})

        if f_shape[-1] is None:
            raise ValueError('The last dimension of the inputs to `CRF` should be defined. Found `None`.')
        if f_shape[-1] != self.output_dim:
            raise ValueError('The last dimension of the input shape must be equal to output shape. '
                             'Use a linear layer if needed.')
        self.input_spec = input_spec
        self.transitions = self.add_weight(name='transitions',
                                           shape=[self.output_dim, self.output_dim],
                                           initializer='glorot_uniform',
                                           trainable=True)
        self.built = True

    def call(self, inputs, **kwargs):
        if self.mode == 'pad':
            sequences = tf.convert_to_tensor(inputs[0], dtype=self.dtype)
            self.sequence_lengths = tf.keras.backend.flatten(inputs[-1])
        else:
            sequences = tf.convert_to_tensor(inputs, dtype=self.dtype)
            shape = tf.shape(inputs)
            self.sequence_lengths = tf.ones(shape[0], dtype=tf.int32) * (shape[1])
        viterbi_sequence, _ = tf.contrib.crf.crf_decode(sequences, self.transitions,
                                                        self.sequence_lengths)
        output = tf.keras.backend.one_hot(viterbi_sequence, self.output_dim)
        return tf.keras.backend.in_train_phase(sequences, output)

    def loss(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred, dtype=self.dtype)
        log_likelihood, self.transitions = tf.contrib.crf.crf_log_likelihood(y_pred,
                                                                             tf.cast(tf.keras.backend.argmax(y_true),
                                                                                     dtype=tf.int32),
                                                                             self.sequence_lengths,
                                                                             transition_params=self.transitions)
        return tf.reduce_mean(-log_likelihood)

    def compute_output_shape(self, input_shape):
        if self.mode == 'pad':
            data_shape = input_shape[0]
        else:
            data_shape = input_shape
        tf.TensorShape(data_shape).assert_has_rank(3)
        return data_shape[:2] + (self.output_dim,)

    @property
    def viterbi_accuracy(self):
        def accuracy(y_true, y_pred):
            shape = tf.shape(y_pred)
            sequence_lengths = tf.ones(shape[0], dtype=tf.int32) * (shape[1])
            viterbi_sequence, _ = tf.contrib.crf.crf_decode(y_pred, self.transitions, sequence_lengths)
            output = tf.keras.backend.one_hot(viterbi_sequence, self.output_dim)
            return tf.keras.metrics.categorical_accuracy(y_true, output)

        accuracy.func_name = 'viterbi_accuracy'
        return accuracy

class LazyOptimizer(Optimizer):
    """Inheriting Optimizer class, wrapping the original optimizer
    to achieve a new corresponding lazy optimizer.
    (Not only LazyAdam, but also LazySGD with momentum if you like.)
    # Arguments
        optimizer: an instance of keras optimizer (supporting
                    all keras optimizers currently available);
        embedding_layers: all Embedding layers you want to update sparsely.
    # Returns
        a new keras optimizer.
    继承Optimizer类，包装原有优化器，实现Lazy版优化器
    （不局限于LazyAdam，任何带动量的优化器都可以有对应的Lazy版）。
    # 参数
        optimizer：优化器实例，支持目前所有的keras优化器；
        embedding_layers：模型中所有你喜欢稀疏更新的Embedding层。
    # 返回
        一个新的keras优化器
    """
    def __init__(self, optimizer, embedding_layers=None, **kwargs):
        super(LazyOptimizer, self).__init__(**kwargs)
        self.optimizer = optimizer
        self.embeddings = []
        if embedding_layers is not None:
            for l in embedding_layers:
                self.embeddings.append(
                    l.trainable_weights[0]
                )
        with K.name_scope(self.__class__.__name__):
            for attr in self.optimizer.get_config():
                if not hasattr(self, attr):
                    value = getattr(self.optimizer, attr)
                    setattr(self, attr, value)
        self.optimizer.get_gradients = self.get_gradients
        self._cache_grads = {}
    def get_gradients(self, loss, params):
        """Cache the gradients to avoiding recalculating.
        把梯度缓存起来，避免重复计算，提高效率。
        """
        _params = []
        for p in params:
            if (loss, p) not in self._cache_grads:
                _params.append(p)
        _grads = super(LazyOptimizer, self).get_gradients(loss, _params)
        for p, g in zip(_params, _grads):
            self._cache_grads[(loss, p)] = g
        return [self._cache_grads[(loss, p)] for p in params]
    def get_updates(self, loss, params):
        # Only for initialization (仅初始化)
        self.optimizer.get_updates(loss, params)
        # Common updates (常规更新)
        dense_params = [p for p in params if p not in self.embeddings]
        self.updates = self.optimizer.get_updates(loss, dense_params)
        # Sparse update (稀疏更新)
        sparse_params = self.embeddings
        sparse_grads = self.get_gradients(loss, sparse_params)
        sparse_flags = [
            K.any(K.not_equal(g, 0), axis=-1, keepdims=True)
            for g in sparse_grads
        ]
        original_lr = self.optimizer.lr
        for f, p in zip(sparse_flags, sparse_params):
            self.optimizer.lr = original_lr * K.cast(f, 'float32')
            # updates only when gradients are not equal to zeros.
            # (gradients are equal to zeros means these words are not sampled very likely.)
            # 仅更新梯度不为0的Embedding（梯度为0意味着这些词很可能是没被采样到的）
            self.updates.extend(
                self.optimizer.get_updates(loss, [p])
            )
        self.optimizer.lr = original_lr
        return self.updates
    def get_config(self):
        config = self.optimizer.get_config()
        return config