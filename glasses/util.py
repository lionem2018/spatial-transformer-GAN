import numpy as np
import imageio
import tensorflow as tf
import os
import termcolor

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# 이미지 로드 함수
def imread(fname):
	return imageio.imread(fname)/255.0


# 이미지 저장 함수
def imsave(fname, array):
	imageio.imsave(fname, (array*255).astype(np.uint8))


# convert to colored strings
# 커맨드 라인에서 글씨에 컬러를 입히기 위한 함수들
def toRed(content): return termcolor.colored(content, "red", attrs=["bold"])


def toGreen(content): return termcolor.colored(content, "green", attrs=["bold"])


def toBlue(content): return termcolor.colored(content, "blue", attrs=["bold"])


def toCyan(content): return termcolor.colored(content, "cyan", attrs=["bold"])


def toYellow(content): return termcolor.colored(content, "yellow", attrs=["bold"])


def toMagenta(content): return termcolor.colored(content, "magenta", attrs=["bold"])


# make image summary from image batch
# 텐서보드를 이용한 학습 과정 시각화를 위해 이미지 서머리(요약) 만드는 함수
def imageSummary(opt, image, tag, H, W):
	blockSize = opt.visBlockSize
	imageOne = tf.batch_to_space(image[:blockSize**2], crops=[[0, 0], [0, 0]], block_size=blockSize)
	imagePermute = tf.reshape(imageOne, [H, blockSize, W, blockSize, -1])
	imageTransp = tf.transpose(imagePermute, [1, 0, 3, 2, 4])
	imageBlocks = tf.reshape(imageTransp, [1, H*blockSize, W*blockSize, -1])
	summary = tf.summary.image(tag, imageBlocks)
	return summary


# restore model
# 저장한 모델을 불러오는 함수
def restoreModelFromIt(opt, sess, saver, net, it):
	saver.restore(sess, os.path.join(SCRIPT_PATH, "models_{0}/{1}_warp{4}_it{2}_{3}.ckpt".format(opt.group, opt.name, it, net, opt.warpN)))


# restore model
# 저장한 모델을 불러오는 함수
def restoreModelPrevStage(opt, sess, saver, net):
	saver.restore(sess, os.path.join(SCRIPT_PATH, "models_{0}/{1}_warp{4}_it{2}_{3}.ckpt".format(opt.group, opt.name, opt.toIt, net, opt.warpN-1)))


# restore model
# 저장한 모델을 불러오는 함수
def restoreModel(opt, sess, saver, path, net):
	saver.restore(sess, os.path.join(SCRIPT_PATH, "models_{0}_{1}.ckpt".format(path, net)))


# save model
# 모델 저장하는 함수
def saveModel(opt, sess, saver, net, it):
	saver.save(sess, os.path.join(SCRIPT_PATH, "models_{0}/{1}_warp{4}_it{2}_{3}.ckpt".format(opt.group, opt.name, it, net, opt.warpN)))

