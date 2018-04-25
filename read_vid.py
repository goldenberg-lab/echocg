import av
import numpy as np
import os
import os.path
import matplotlib.pyplot as plt
import csv
import tensorflow as tf
from skimage.transform import resize
import scipy.ndimage
from skimage import img_as_float



def read_video(path_to_file):
	''' read a single video file and return a numpy
	array with all frames in shape : [num frames, image height,
	image weight, channels].
	'''
	container = av.open(path_to_file)
	frames = []
	for frame in container.decode(video=0):
		img = frame.to_image()
		arr = np.asarray(img, dtype=np.ubyte)
		frames.append(arr)
		
	frames = np.stack(frames, axis=0)
	return frames


def read_all_subj(dirc):
	''' Directory -> {subj_id:[videos]}
	Read all videos for all subjects. dirc is parent directory
	which contains subject directories.Each subject directory contains
	containes videos for that subject.
	'''
	subj_echo = dict()
	subj_dirs = os.listdir(dirc)
	for subj_dir in subj_dirs:
		subj_files = os.listdir(os.path.join(dirc, subj_dir))
		subj_videos = []
		for video_file in subj_files:	
			frames = read_video(os.path.join(dirc,
				subj_dir, video_file))
			subj_videos.append(frames)
		subj_echo[subj_dir] = subj_videos
	return subj_echo	


def read_clinical_csv(path):
	''' str -> {ptid: {feature: value}}
	Return a dictionary indexed by ptid that containes clinical
	information for each patient.
	'''
	clin_data = dict()
	with open(path) as clin_file:
		reader = csv.DictReader(clin_file)
		for example in reader:
			ptid = example['ptid']
			prim_out = int(example['primout'])-1
			clin_data[ptid]={'prim_out':prim_out}
	return clin_data


def write_TF_records(path, subj_echo, clin_data):
	with tf.python_io.TFRecordWriter(path) as writer:
		for ptid in subj_echo.keys():
			for video in subj_echo[ptid]:
				outcome = clin_data[ptid]['prim_out']
				for i in range(video.shape[0]//120):
					example = make_seq_example(ptid,
					video[i*120:(i+1)*120,:,:,:], outcome)
					writer.write(example.SerializeToString())
				remaining = video.shape[0]%120
				if (remaining):
					example = make_seq_example(ptid,
						video[-1*remaining:,:,:,:],outcome)
					writer.write(example.SerializeToString())
					

def make_seq_example(ptid, video, outcome):
	example = tf.train.SequenceExample()
	example.context.feature['ptid'].bytes_list.value.append(ptid.encode())
	example.context.feature['primout'].int64_list.value.append(outcome)

	frames = example.feature_lists.feature_list['frames']
	for i in range(video.shape[0]):
		processed_frame = process_image(video[i],3)
		assert(not np.isnan(processed_frame).any())
		# print("processed", processed_frame.shape, processed_frame.dtype)
		frames.feature.add().bytes_list.value.append(processed_frame.tostring())
	return example


def parse_example(example):
	context_feature = {
		'ptid':tf.FixedLenFeature([], tf.string),
		'primout':tf.FixedLenFeature([], tf.int64)}
	sequence_feature = {'frames': tf.FixedLenSequenceFeature([], tf.string)}
	
	context_parsed, sequence_parsed = tf.parse_single_sequence_example(example, context_features=context_feature, sequence_features=sequence_feature)
	ptid = tf.cast(context_parsed['ptid'], tf.string)
	primout = tf.cast(context_parsed['primout'], tf.int64)
	frames = tf.decode_raw(sequence_parsed['frames'], tf.float64)
	frames = tf.cast(frames, tf.float32)
	frames = tf.reshape(frames, [-1, 224, 224, 3])
	return (tf.expand_dims(ptid, axis=0),
		tf.expand_dims(primout, axis=0),
		frames)		


def process_image(image, sigma):
	image = img_as_float(image)
	image = scipy.ndimage.filters.gaussian_filter(image, sigma)
	image = image[:,80:560,:]
	image = resize(image, (224, 224))
	return image




if (__name__=="__main__"):
	BASE_PATH = "/home/yasaman/echo/BV/"	
	CLIN_PATH = "/home/yasaman/echo/fetal_coarctation_data.csv"
	REC_PATH = "/home/yasaman/echo/subj_video.tfrecords"
	subj_echo = read_all_subj(BASE_PATH)
	clin_data = read_clinical_csv(CLIN_PATH)
	subj = subj_echo['w9ttsx']
	image = subj[0][0]
	processed_image = process_image(image, 1)
	plt.figure()
	plt.imshow(image)
	plt.figure()
	plt.imshow(processed_image)
	# check image processing 
	#plt.show()
		
	write_TF_records(REC_PATH, subj_echo, clin_data)









