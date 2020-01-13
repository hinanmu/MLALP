#@Time      :2019/9/26 21:07
#@Author    :zhounan
#@FileName  :evaluate.py
# loss function for tensorflow
import tensorflow as tf

def hamming_loss(ground_truth, output):
    prediction = tf.to_float(tf.greater_equal(output, 0.5))
    num_instances = tf.shape(output)[0]
    num_labels = tf.shape(output)[1]

    hamming_loss = tf.reduce_sum(tf.to_float(tf.not_equal(prediction, ground_truth))) / tf.to_float(num_labels * num_instances)
    return hamming_loss

def pairwise_and(a, b):
    """compute pairwise logical and between elements of the tensors a and b
    Description
    -----
    if y shape is [3,3], y_i would be translate to [3,3,1], y_not_i is would be [3,1,3]
    and return [3,3,3],through the matrix ,we can easy to caculate c_k - c_i(appear in the paper)
    """
    column = tf.expand_dims(a, 2)
    row = tf.expand_dims(b, 1)
    return tf.logical_and(column, row)

def pairwise_sub(a, b):
    """compute pairwise differences between elements of the tensors a and b
    :param a:
    :param b:
    :return:
    """
    column = tf.expand_dims(a, 2)
    row = tf.expand_dims(b, 1)
    return tf.subtract(column, row)

def ranking_loss(ground_truth, output):
    num_instances = tf.shape(output)[0]
    num_labels = tf.shape(output)[1]

    y_i = tf.equal(ground_truth, tf.ones_like(ground_truth))
    y_not_i = tf.equal(ground_truth, tf.zeros_like(ground_truth))

    truth_matrix = tf.to_float(pairwise_and(y_i, y_not_i))

    # calculate all exp'd differences
    # through and with truth_matrix, we can get all c_i - c_k(appear in the paper)
    sub_matrix = pairwise_sub(output, output)
    sparse_matrix = tf.multiply(tf.to_float(tf.less_equal(sub_matrix, 0)), truth_matrix)

    sums = tf.reduce_sum(sparse_matrix, axis=[1, 2])

    # get normalizing terms and apply them
    y_i_sizes = tf.reduce_sum(tf.to_float(y_i), axis=1)
    y_i_bar_sizes = tf.reduce_sum(tf.to_float(y_not_i), axis=1)
    normalizers = tf.multiply(y_i_sizes, y_i_bar_sizes)
    normalizers_zero = tf.equal(normalizers, tf.zeros_like(normalizers))
    normalizers = tf.where(normalizers_zero, tf.ones_like(normalizers), normalizers)
    sums = tf.where(normalizers_zero, tf.zeros_like(sums), sums)
    loss = tf.divide(sums, normalizers)
    ranking_loss = tf.reduce_sum(loss) / (tf.to_float(tf.shape(loss)[0]) - tf.reduce_sum(tf.to_float(normalizers_zero)))

    return ranking_loss

def average_precision(ground_truth, output):
    num_instances = tf.shape(output)[0]
    num_labels = tf.shape(output)[1]
    sort_output, idx = tf.nn.top_k(output, num_labels)
    idx_1 = tf.stack([tf.tile(tf.reshape(tf.range(num_instances), shape=(-1, 1)), multiples=[1, num_labels]), idx],
                     axis=-1)
    sort_ground_truth = tf.gather_nd(ground_truth, idx_1)

    rank = tf.cumsum(sort_ground_truth, axis=1)

    range_ = tf.to_float(tf.tile(tf.reshape(tf.range(num_labels), shape=(1, -1)), multiples=[num_instances, 1]))

    precision = tf.reduce_sum(sort_ground_truth * rank / (range_ + 1), axis=1)
    norm = tf.reduce_sum(sort_ground_truth, axis=1)

    norm_zero = tf.equal(norm, tf.zeros_like(norm))
    norm = tf.where(norm_zero, tf.ones_like(norm), norm)
    precision = tf.where(norm_zero, tf.zeros_like(precision), precision)
    ap = tf.divide(precision , norm)
    ap = tf.reduce_sum(ap) / (tf.to_float(tf.shape(ap)[0]) - tf.reduce_sum(tf.to_float(norm_zero)))


    return ap

def micro_f1(ground_truth, output):
    prediction = tf.to_float(tf.greater_equal(output, 0.5))

    epsilon = 1e-7
    tp = tf.reduce_sum(prediction * ground_truth, axis=0)
    # tn = tf.sum(tf.cast((1-y_hat)*(1-y_true), 'float'), axis=0)
    fp = tf.reduce_sum((1 - prediction) * ground_truth, axis=0)
    fn = tf.reduce_sum(prediction * (1 - ground_truth), axis=0)

    p = tp/(tp+fp+epsilon)#epsilon的意义在于防止分母为0，否则当分母为0时python会报错
    r = tp/(tp+fn+epsilon)

    f1 = 2 * p * r / (p + r + epsilon)
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return tf.reduce_mean(f1)

def macro_f1(ground_truth, output):
    prediction = tf.to_float(tf.greater_equal(output, 0.5))

    epsilon = 1e-7
    tp = tf.reduce_sum(prediction * ground_truth, axis=0)
    # tn = tf.sum(tf.cast((1-y_hat)*(1-y_true), 'float'), axis=0)
    fp = tf.reduce_sum((1 - prediction) * ground_truth, axis=0)
    fn = tf.reduce_sum(prediction * (1 - ground_truth), axis=0)

    p = tf.reduce_sum(tp) / (tf.reduce_sum(tp) + tf.reduce_sum(fp))
    r = tf.reduce_sum(tp) / (tf.reduce_sum(tp) + tf.reduce_sum(fn))
    f1 = 2 * (p * r) / (p + r + epsilon)
    return f1