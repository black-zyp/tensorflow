import numpy as np


# 第一步:计算 loss 函数    y = w*x + b
def computer_error_for_line_given_points(b, w, points):
    total_Error = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total_Error += ((w * x + b) - y) ** 2
    return total_Error / float(len(points))


# 第二部:完成梯度的计算和更新
def step_gradient(b_current, w_current, points, learningRate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # grad_b = 2(wx+b) b的梯度
        b_gradient += (2 / N) * ((w_current * x + b_current) - y)
        # grad_w = 2(wx+b)+x w的梯度
        w_gradient += (2 / N) * x * ((w_current * x + b_current) - y)

    # update W'     learningRate 衰减率
    new_b = b_current - (learningRate * b_gradient)
    new_w = w_current - (learningRate * w_gradient)
    return [new_b, new_w]


# 第三步:循环计算得出最好的 b 和 w
def gradient_descent_runner(points, starting_b, starting_w, learning_rate, num_iterations):
    b = starting_b
    w = starting_w
    for i in range(num_iterations):
        b, w = step_gradient(b, w, np.array(points), learning_rate)
    return [b, w]


# 运行程序
def run():
    points = np.genfromtxt("../data_file/data.csv", delimiter=",")
    print(type(points))
    learning_rate = 0.0001  # 学习率
    initial_b = 0  # 设置最初的 b
    initial_w = 0  # 设置最初的 w
    initial_error = computer_error_for_line_given_points(initial_b, initial_w, points)  # 经过计算得出最初的
    num_iterations = 1000  # 设置循环计算的次数
    print("Starting gradient descent at b = {0}, w = {1}, error = {2}"
          .format(initial_b, initial_w, initial_error)
          )
    print("Running...")
    [b, w] = gradient_descent_runner(points, initial_b, initial_w, learning_rate, num_iterations)
    end_error = computer_error_for_line_given_points(b, w, points)
    print("After {0} iterations b = {1}, w = {2}, error = {3}"
          .format(num_iterations, b, w, end_error)
          )


if __name__ == '__main__':
    run()
