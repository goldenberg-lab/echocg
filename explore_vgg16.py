import numpy as np
import tensorflow as tf
from read_vid import *
import tensorflow.contrib.slim.nets as nets
from sklearn.manifold import TSNE


''' plot projections of features pretrained vgg16 extracts from echocardio grams.'''



IMG_NET_WEIGHT_PATH = '/home/yasaman/HN/image_net_trained/vgg_16.ckpt'
RECORD_PATH = "/home/yasaman/echo/subj_video.tfrecords"

vgg=nets.vgg

input_files = tf.placeholder(tf.string, shape=None)

dataset = tf.data.TFRecordDataset(input_files)
dataset = dataset.map(parse_example)
dataset = dataset.shuffle(buffer_size=1000)

iterator = dataset.make_initializable_iterator()


next_ptid, next_outcome, next_frames = iterator.get_next()


outcome_onehot = tf.one_hot(next_outcome, depth=3, dtype=tf.int64)

logits, intermed = vgg.vgg_16(next_frames, is_training=False, spatial_squeeze=False)


fc7 = intermed['vgg_16/fc7']
fc7 = tf.squeeze(fc7)

restorer = tf.train.Saver()

features = []
labels  = []

with tf.Session() as sess:

	restorer.restore(sess, IMG_NET_WEIGHT_PATH)
	sess.run(iterator.initializer, feed_dict={input_files:RECORD_PATH})
	for i in range(30):
#	print(np.argwhere(np.isnan(frames[0])), frames[0].min(), frames[0].max())
		extracted_features, label, ptid = sess.run((fc7, next_outcome, next_ptid))
		print("label", label, "ptid", ptid, extracted_features.shape[0])
		features.append(extracted_features)
		labels.append(label * np.ones(extracted_features.shape[0]))




all_frames = np.concatenate(features, axis=0)
all_labels = np.concatenate(labels, axis=0)


embedded_frames = TSNE().fit_transform(all_frames)
fig = plt.figure()
ax1 = plt.subplot(111)
ax1.plot(embedded_frames[all_labels==0,0], embedded_frames[all_labels==0,1], 'o')
ax1.plot(embedded_frames[all_labels==1,0], embedded_frames[all_labels==1,1], 'ro')
ax1.plot(embedded_frames[all_labels==2,0], embedded_frames[all_labels==2,1], 'ko')
plt.show()

