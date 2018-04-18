import av
import numpy as np
import os
import os.path
import matplotlib.pyplot as plt
import csv
import tensorflow as tf


def read_video(path_to_file):
	''' read a single video file and return a numpy
	array with all frames in shape : [num frames, image height,
	image weight, channels].
	'''
	container = av.open(path_to_file)
	frames = []
	for frame in container.decode(video=0):
		img = frame.to_image()
		arr = np.asarray(img)
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
				example = make_seq_example(ptid, video, outcome)
				writer.write(example.SerializeToString())


def make_seq_example(ptid, video, outcome):
	example = tf.train.SequenceExample()
	example.context.feature['ptid'].bytes_list.value.append(ptid.encode())
	example.context.feature['primout'].int64_list.value.append(outcome)

	frames = example.feature_lists.feature_list['frames']
	for i in range(video.shape[0]):
		frames.feature.add().bytes_list.value.append(video[i].tostring())
	return example




if (__name__=="__main__"):
	BASE_PATH = "/home/yasaman/echo/BV/"	
	CLIN_PATH = "/home/yasaman/echo/fetal_coarctation_data.csv"
	REC_PATH =  "/home/yasaman/echo/subj_video.tfrecords"
	subj_echo = read_all_subj(BASE_PATH)
	clin_data = read_clinical_csv(CLIN_PATH)
	
	write_TF_records(REC_PATH, subj_echo, clin_data)









