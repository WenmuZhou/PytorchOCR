import cv2
import abc
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw


def cv2pil(image):
    """
    将bgr格式的numpy的图像转换为pil
    :param image:   图像数组
    :return:    Image对象
    """
    assert isinstance(image, np.ndarray), 'input image type is not cv2'
    if len(image.shape) == 2:
        return Image.fromarray(image)
    elif len(image.shape) == 3:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def get_pil_image(image):
    """
    将图像统一转换为PIL格式
    :param image:   图像
    :return:    Image格式的图像
    """
    if isinstance(image, Image.Image):  # or isinstance(image, PIL.JpegImagePlugin.JpegImageFile):
        return image
    elif isinstance(image, np.ndarray):
        return cv2pil(image)


def get_cv_image(image):
    """
    将图像转换为numpy格式的数据
    :param image:   图像
    :return:    ndarray格式的图像数据
    """
    if isinstance(image, np.ndarray):
        return image
    elif isinstance(image, Image.Image):  # or isinstance(image, PIL.JpegImagePlugin.JpegImageFile):
        return pil2cv(image)


def pil2cv(image):
    """
    将Image对象转换为ndarray格式图像
    :param image:   图像对象
    :return:    ndarray图像数组
    """
    if len(image.split()) == 1:
        return np.asarray(image)
    elif len(image.split()) == 3:
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    elif len(image.split()) == 4:
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGBA2BGR)


class TransBase(object):
    """
    数据增广的基类
    """

    def __init__(self, probability=1.):
        """
        初始化对象
        :param probability:     执行概率
        """
        super(TransBase, self).__init__()
        self.probability = probability

    @abc.abstractmethod
    def trans_function(self, _image):
        """
        初始化执行函数，需要进行重载
        :param _image:  待处理图像
        :return:    执行后的Image对象
        """
        pass

    # @utils.zlog
    def process(self, _image):
        """
        调用执行函数
        :param _image:  待处理图像
        :return:    执行后的Image对象
        """
        if np.random.random() < self.probability:
            return self.trans_function(_image)
        else:
            return _image

    def __call__(self, _image):
        """
        重载()，方便直接进行调用
        :param _image:  待处理图像
        :return:    执行后的Image
        """
        return self.process(_image)


class RandomContrast(TransBase):
    """
    随机对比度
    """

    def setparam(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "upper must be >= lower."
        assert self.lower >= 0, "lower must be non-negative."

    def trans_function(self, _image):
        _image = get_pil_image(_image)
        contrast_enhance = ImageEnhance.Contrast(_image)
        return contrast_enhance.enhance(random.uniform(self.lower, self.upper))


class RandomLine(TransBase):
    """
    在图像增加一条简单的随机线
    """

    def trans_function(self, image):
        image = get_pil_image(image)
        draw = ImageDraw.Draw(image)
        h = image.height
        w = image.width
        y0 = random.randint(h // 4, h * 3 // 4)
        y1 = np.clip(random.randint(-3, 3) + y0, 0, h - 1)
        color = random.randint(0, 30)
        draw.line(((0, y0), (w - 1, y1)), fill=(color, color, color), width=2)
        return image


class RandomBrightness(TransBase):
    """
    随机对比度
    """

    def setparam(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "upper must be >= lower."
        assert self.lower >= 0, "lower must be non-negative."

    def trans_function(self, image):
        image = get_pil_image(image)
        bri = ImageEnhance.Brightness(image)
        return bri.enhance(random.uniform(self.lower, self.upper))


class RandomColor(TransBase):
    """
    随机色彩平衡
    """

    def setparam(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "upper must be >= lower."
        assert self.lower >= 0, "lower must be non-negative."

    def trans_function(self, image):
        image = get_pil_image(image)
        col = ImageEnhance.Color(image)
        return col.enhance(random.uniform(self.lower, self.upper))


class RandomSharpness(TransBase):
    """
    随机锐度
    """

    def setparam(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "upper must be >= lower."
        assert self.lower >= 0, "lower must be non-negative."

    def trans_function(self, image):
        image = get_pil_image(image)
        sha = ImageEnhance.Sharpness(image)
        return sha.enhance(random.uniform(self.lower, self.upper))


class Compress(TransBase):
    """
    随机压缩率，利用jpeg的有损压缩来增广
    """

    def setparam(self, lower=5, upper=85):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "upper must be >= lower."
        assert self.lower >= 0, "lower must be non-negative."

    def trans_function(self, image):
        img = get_cv_image(image)
        param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(self.lower, self.upper)]
        img_encode = cv2.imencode('.jpeg', img, param)
        img_decode = cv2.imdecode(img_encode[1], cv2.IMREAD_COLOR)
        pil_img = cv2pil(img_decode)
        if len(image.split()) == 1:
            pil_img = pil_img.convert('L')
        return pil_img


class Exposure(TransBase):
    """
    随机区域曝光
    """

    def setparam(self, lower=5, upper=10):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "upper must be >= lower."
        assert self.lower >= 0, "lower must be non-negative."

    def trans_function(self, image):
        image = get_cv_image(image)
        h, w = image.shape[:2]
        x0 = random.randint(0, w)
        y0 = random.randint(0, h)
        x1 = random.randint(x0, w)
        y1 = random.randint(y0, h)
        transparent_area = (x0, y0, x1, y1)
        mask = Image.new('L', (w, h), color=255)
        draw = ImageDraw.Draw(mask)
        mask = np.array(mask)
        if len(image.shape) == 3:
            mask = mask[:, :, np.newaxis]
            mask = np.concatenate([mask, mask, mask], axis=2)
        draw.rectangle(transparent_area, fill=random.randint(150, 255))
        reflection_result = image + (255 - mask)
        reflection_result = np.clip(reflection_result, 0, 255)
        return cv2pil(reflection_result)


class Rotate(TransBase):
    """
    随机旋转
    """

    def setparam(self, lower=-5, upper=5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "upper must be >= lower."

    def trans_function(self, image):
        image = get_pil_image(image)
        rot = random.uniform(self.lower, self.upper)
        trans_img = image.rotate(rot, expand=True)
        return trans_img


class Blur(TransBase):
    """
    随机高斯模糊
    """

    def setparam(self, lower=0, upper=1):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "upper must be >= lower."
        assert self.lower >= 0, "lower must be non-negative."

    def trans_function(self, image):
        image = get_pil_image(image)
        image = image.filter(ImageFilter.GaussianBlur(radius=1.5))
        return image


class MotionBlur(TransBase):
    """
    随机运动模糊
    """

    def setparam(self, degree=5, angle=180):
        self.degree = degree
        self.angle = angle

    def trans_function(self, image):
        image = get_pil_image(image)
        angle = random.randint(0, self.angle)
        M = cv2.getRotationMatrix2D((self.degree / 2, self.degree / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(self.degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (self.degree, self.degree))
        motion_blur_kernel = motion_blur_kernel / self.degree
        image = image.filter(ImageFilter.Kernel(size=(self.degree, self.degree), kernel=motion_blur_kernel.reshape(-1)))
        return image


class Salt(TransBase):
    """
    随机椒盐噪音
    """

    def setparam(self, rate=0.02):
        self.rate = rate

    def trans_function(self, image):
        image = get_pil_image(image)
        num_noise = int(image.size[1] * image.size[0] * self.rate)
        # assert len(image.split()) == 1
        for k in range(num_noise):
            i = int(np.random.random() * image.size[1])
            j = int(np.random.random() * image.size[0])
            image.putpixel((j, i), int(np.random.random() * 255))
        return image


class AdjustResolution(TransBase):
    """
    随机分辨率
    """

    def setparam(self, max_rate=0.95, min_rate=0.5):
        self.max_rate = max_rate
        self.min_rate = min_rate

    def trans_function(self, image):
        image = get_pil_image(image)
        w, h = image.size
        rate = np.random.random() * (self.max_rate - self.min_rate) + self.min_rate
        w2 = int(w * rate)
        h2 = int(h * rate)
        image = image.resize((w2, h2))
        image = image.resize((w, h))
        return image


class Crop(TransBase):
    """
    随机抠图，并且抠图区域透视变换为原图大小
    """

    def setparam(self, maxv=2):
        self.maxv = maxv

    def trans_function(self, image):
        img = get_cv_image(image)
        h, w = img.shape[:2]
        org = np.array([[0, np.random.randint(0, self.maxv)],
                        [w, np.random.randint(0, self.maxv)],
                        [0, h - np.random.randint(0, self.maxv)],
                        [w, h - np.random.randint(0, self.maxv)]], np.float32)
        dst = np.array([[0, 0], [w, 0], [0, h], [w, h]], np.float32)
        M = cv2.getPerspectiveTransform(org, dst)
        res = cv2.warpPerspective(img, M, (w, h))
        return get_pil_image(res)


class Crop2(TransBase):
    """
    随机抠图，并且抠图区域透视变换为原图大小
    """

    def setparam(self, maxv_h=4, maxv_w=4):
        self.maxv_h = maxv_h
        self.maxv_w = maxv_w

    def trans_function(self, image_and_loc):
        image, left, top, right, bottom = image_and_loc
        w, h = image.size
        left = np.clip(left, 0, w - 1)
        right = np.clip(right, 0, w - 1)
        top = np.clip(top, 0, h - 1)
        bottom = np.clip(bottom, 0, h - 1)
        img = get_cv_image(image)
        try:
            res = get_pil_image(img[top:bottom, left:right])
            return res
        except AttributeError as e:
            print('error')
            image.save('test_imgs/t.png')
            print(left, top, right, bottom)

        h = bottom - top
        w = right - left
        org = np.array(
            [[left - np.random.randint(0, self.maxv_w), top + np.random.randint(-self.maxv_h, self.maxv_h // 2)],
             [right + np.random.randint(0, self.maxv_w), top + np.random.randint(-self.maxv_h, self.maxv_h // 2)],
             [left - np.random.randint(0, self.maxv_w), bottom - np.random.randint(-self.maxv_h, self.maxv_h // 2)],
             [right + np.random.randint(0, self.maxv_w), bottom - np.random.randint(-self.maxv_h, self.maxv_h // 2)]],
            np.float32)
        dst = np.array([[0, 0], [w, 0], [0, h], [w, h]], np.float32)
        M = cv2.getPerspectiveTransform(org, dst)
        res = cv2.warpPerspective(img, M, (w, h))
        return get_pil_image(res)


class Stretch(TransBase):
    """
    随机图像横向拉伸
    """

    def setparam(self, max_rate=1.2, min_rate=0.8):
        self.max_rate = max_rate
        self.min_rate = min_rate

    def trans_function(self, image):
        image = get_pil_image(image)
        w, h = image.size
        rate = np.random.random() * (self.max_rate - self.min_rate) + self.min_rate
        w2 = int(w * rate)
        image = image.resize((w2, h))
        return image


class DataAug:
    def __init__(self):
        self.crop = Crop(probability=0.1)
        self.crop2 = Crop2(probability=1.1)
        self.random_contrast = RandomContrast(probability=0.1)
        self.random_brightness = RandomBrightness(probability=0.1)
        self.random_color = RandomColor(probability=0.1)
        self.random_sharpness = RandomSharpness(probability=0.1)
        self.compress = Compress(probability=0.3)
        self.exposure = Exposure(probability=0.1)
        self.rotate = Rotate(probability=0.1)
        self.blur = Blur(probability=0.3)
        self.motion_blur = MotionBlur(probability=0.3)
        self.salt = Salt(probability=0.1)
        self.adjust_resolution = AdjustResolution(probability=0.1)
        self.stretch = Stretch(probability=0.1)
        self.random_line = RandomLine(probability=0.3)
        self.crop.setparam()
        self.crop2.setparam()
        self.random_contrast.setparam()
        self.random_brightness.setparam()
        self.random_color.setparam()
        self.random_sharpness.setparam()
        self.compress.setparam()
        self.exposure.setparam()
        self.rotate.setparam()
        self.blur.setparam()
        self.motion_blur.setparam()
        self.salt.setparam()
        self.adjust_resolution.setparam()
        self.stretch.setparam()

    def aug_img(self, img):
        img = self.crop.process(img)
        img = self.random_contrast.process(img)
        img = self.random_brightness.process(img)
        img = self.random_color.process(img)
        img = self.random_sharpness.process(img)
        img = self.random_line.process(img)

        if img.size[1] >= 32:
            img = self.compress.process(img)
            img = self.adjust_resolution.process(img)
            img = self.motion_blur.process(img)
            img = self.blur.process(img)
        img = self.exposure.process(img)
        img = self.rotate.process(img)
        img = self.salt.process(img)
        img = self.inverse_color(img)
        img = self.stretch.process(img)
        return img

    def inverse_color(self, image):
        if np.random.random() < 0.4:
            image = ImageOps.invert(image)
        return image


if __name__ == '__main__':
    img = Image.open('./test_image.png')
    data_augment = DataAug()
    augmented_img = data_augment.aug_img(img)
    augmented_img.show()