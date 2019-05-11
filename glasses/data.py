import numpy as np
import tensorflow as tf
import os,time
import warp


def load(opt, test=False):
	"""
	load data
	데이터 불러오는 함수
	:param opt: 옵션
	:param test: 테스트를 위해 데이터를 불러오는지
	:return: 데이터셋이 담긴 디렉토리
	"""
	# 불러올 디렉토리 경로 설정
	main_folder = "/content/gdrive/My Drive/Colab Notebooks/spatial-transformer-GAN/glasses/"
	path = main_folder + "dataset"
	# 테스트 데이터셋를 불러와야 한다면,
	# if test:
		# 테스트 이미지 데이터가 담긴 .npy 파일 불러오기
	images_0 = np.load("{0}/not_wearing_earring_test.npy".format(path))    # 귀걸이 안 낀 이미지 npy 파일
	images_1 = np.load("{0}/earring_image_test.npy".format(path))  # 귀걸이 낀 이미지 npy 파일
	# 	# 귀걸이 착용 유무 속성정보 불러오기
	# hasGlasses = np.load("{0}/attribute_test.npy".format(path))[:, 35]

	# # 학습 데이터셋을 불러와야 한다면,
	# else:
	# 	# 학습 이미지 데이터가 담긴 .npy 파일 불러오기
	# 	images = np.load("{0}/image_train.npy".format(path))
	# 	# 안경 유무 속성 정보 불러오기
	# 	hasGlasses = np.load("{0}/attribute_train.npy".format(path))[:, 15]

	# # 안경 유무에 따라 이미지 데이터셋 나누기
	# images_0 = images[~hasGlasses]
	# images_1 = images[hasGlasses]

	# 안경 데이터셋 불러오기
	glasses = np.load("{0}/earring.npy".format(path))

	# 디렉토리 생성
	D = {
		"image0": images_0,
		"image1": images_1,
		"glasses": glasses,
	}
	return D


def makeBatch(opt, data, PH):
	"""
	make training batch
	학습을 위한 배치(학습 데이터 덩어리) 만드는 함수
	:param opt: 옵션
	:param data: 배치를 가져올 전체 데이터셋
	:param PH: placeholder
	:return: 배치 데이터
	"""
	# 데이터셋에서 각 데이터들의 개수를 구함
	N0 = len(data["image0"])
	N1 = len(data["image1"])
	NG = len(data["glasses"])

	# 배치 사이즈 만큼의 데이터 인덱스를 배치 사이즈 만큼 생성
	randIdx0 = np.random.randint(N0, size=[opt.batchSize])
	randIdx1 = np.random.randint(N1, size=[opt.batchSize])
	randIdxG = np.random.randint(NG, size=[opt.batchSize])

	# put data in placeholders
	[imageBGfakeData, imageRealData, imageFGfake] = PH

	# 각 픽셀이 0~255로 되어있는 상태를 0~1로 전처리해주며 배치 사이즈만큼 가져옴
	batch = {
		imageBGfakeData: data["image0"][randIdx0]/255.0,
		imageRealData: data["image1"][randIdx1]/255.0,
		imageFGfake: data["glasses"][randIdxG]/255.0,
	}
	return batch


# make test batch
def makeBatchEval(opt, testImage, glasses, PH):
	"""
	make test batch
	테스트를 위한 배치 만드는 함수
	:param opt: 옵션
	:param testImage: 테스트할 인물 이미지 데이터셋
	:param glasses: 안경 이미지 데이터셋
	:param PH: placeholder
	:return: 배치 데이터
	"""
	# 배치 사이즈만큼의 1차원 배열 생성 0, 1, 2, ..., batchSize-1
	idxG = np.arange(opt.batchSize)

	print(idxG)

	# put data in placeholders
	[imageBG, imageFG] = PH

	batch = {
		# 인물 이미지 배치 사이즈 만큼 반복하여 연결
		imageBG: np.tile(testImage, [opt.batchSize, 1, 1, 1]),
		# 배치 사이즈 만큼의 안경 데이터 가져옴
		imageFG: glasses[idxG]/255.0,
	}
	return batch


def perturbBG(opt, imageData):
	"""
	generate pereturbed image
	교란된 이미지 생성하는 함수
	:param opt: 옵션
	:param imageData: 전체 이미지 데이터셋
	:return: 이미지 데이터
	"""
	# batch 사이즈 만큼의 랜덤한 값을 가진 배열들을 생성하여 각 원소에 pertBG 값 곱해줌
	rot = opt.pertBG*tf.random_normal([opt.batchSize])
	tx = opt.pertBG*tf.random_normal([opt.batchSize])
	ty = opt.pertBG*tf.random_normal([opt.batchSize])
	# 배치 사이즈 만큼의 0으로만 이루어진 배열 생성
	O = tf.zeros([opt.batchSize])
	# 교란된 백그라운드 이미지(인물 사진) 생성을 위한 warp 매개변수 생성
	# 왜곡 타입이 homography(가로, 세로간의 평행을 유지하지 않고 이미지를 왜곡시킴)일 때와
	# 왜곡 타입이 affine(가로, 세로간의 평행을 유지하며 이미지를 왜곡시킴)으로 나누어 생성
	pPertBG = \
		tf.stack([tx, rot, O, O, ty, -rot, O, O], axis=1) if opt.warpType == "homography" else\
		tf.stack([O, rot, tx, -rot, O, ty], axis=1) if opt.warpType == "affine" else None
	# 벡터 형태의 warp 매개변수를 matrix로 변환
	pPertBGmtrx = warp.vec2mtrx(opt, pPertBG)
	# warp 파라미터에 따라 이미지 데이터셋을 변형 및 크롭하여 생성
	image = warp.transformCropImage(opt, imageData, pPertBGmtrx)
	return image


history = [None, 0, True]


# update history and group fake samples
def updateHistory(opt, newFake):
	"""
	히스토리를 업데이트하고 가짜 샘플을 그룹핑하는 함수
	:param opt: 옵션들
	:param newFake: 새로운 가짜 이미지
	:return: 페이크 이미지
	"""
	if history[0] is None:
		history[0] = np.ones([opt.histQsize, opt.H, opt.W, 3], dtype=np.float32)
		history[0][:opt.batchSize] = newFake
		history[1] = opt.batchSize
		return newFake
	else:
		randIdx = np.random.permutation(opt.batchSize)
		storeIdx = randIdx[:opt.histSize]
		useIdx = randIdx[opt.histSize:]
		# group fake samples
		hi, growing = history[1], history[2]
		extractIdx = np.random.permutation(hi if growing else opt.histQsize)[:opt.histSize]
		groupFake = np.concatenate([history[0][extractIdx], newFake[useIdx]], axis=0)
		hinew = hi+opt.batchSize-opt.histSize
		history[0][hi:hinew] = newFake[storeIdx]
		history[1] = hinew
		if hinew == opt.histQsize:
			history[1] = 0
			history[2] = False
		return groupFake
