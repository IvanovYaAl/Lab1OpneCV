import numpy as np
import cv2

def viewImage(image, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_window, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def convolution(data: np.ndarray, kernels: np.ndarray) -> np.ndarray:
    data_channels, data_height, data_width = data.shape
    kernels_count, kernel_channels, kernel_height, kernel_width = kernels.shape
    result = np.zeros(shape=(kernels_count, data_height, data_width))

    print("h: ", kernel_height," w: ", kernel_width)
    print("kernels: ", kernels_count, " data_height: ", data_height," data_width: ", data_width)
    assert (data_channels == kernel_channels)
    for kernel in range(kernels_count):
        for y in range(1, data_height - 1):
            for x in range(1, data_width - 1):
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        for k in range(data_channels):
                            result[kernel, y, x] += data[k, y + i, x + j] * kernels[kernel, k, i, j]
    return result


def normalize(data, beta, gamma):
    result = np.zeros(shape=data.shape)
    data_channels, data_height, data_width = data.shape

    mu = np.average(data, axis=(1, 2))
    standard_deviation = np.std(data, axis=(1, 2))
    for channel in range(data_channels):
        result[channel, :, :] = (data[channel, :, :] - mu[channel]) / (standard_deviation[channel] + 1e-12)
        result[channel, :, :] = gamma[channel, :, :] * result[channel, :, :] + beta[channel, :, :]
    return result


def max_pooling(data):
    data_channels, data_height, data_width = data.shape

    kernel_height = 2
    kernel_width = 2

    result_height = data_height // kernel_height
    result_width = data_width // kernel_width

    result = np.zeros(shape=(data_channels, result_height, result_width))

    for channel in range(data_channels):
        for y in range(result_height):
            for x in range(result_width):
                y_from = y * kernel_height
                y_to = y * kernel_height + kernel_height
                x_from = x * kernel_width
                x_to = x * kernel_width + kernel_width
                result[channel, y, x] = np.amax(data[channel, y_from:y_to, x_from:x_to], axis=(0, 1))
    return result


def softmax(data: np.ndarray) -> np.ndarray:
    data_height, data_width, _ = data.shape
    output_layer = np.zeros(shape=data.shape)

    for h in range(data_height):
        for w in range(data_width):
            output_layer[h, w, :] = np.exp(data[h, w]) / sum(np.exp(data[h, w]))
    return output_layer


if __name__ == '__main__':
    
    image = cv2.imread('C://Users//yaros//.spyder-py3//Bots//Cat2.png')
    data = np.transpose(image, (2, 1, 0))
    viewImage(image, "Cat")
    print("Converted shape ",'Shape: {}'.format(data.shape))
    
    data = convolution(data, np.random.rand(5, 3, 3, 3))
    print("Convolution shape ",'Shape: {}'.format(data.shape))

    data = normalize(data, np.random.uniform(2, 8, data.shape), np.random.uniform(2, 8, data.shape))
    print("Normalize shape ",'Shape: {}'.format(data.shape))

    data = max_pooling(data)
    print("Max pooling shape ",'Shape: {}'.format(data.shape))

    data = softmax(data)
    print("Prediction shape ",'Shape: {}'.format(data.shape))
    
    