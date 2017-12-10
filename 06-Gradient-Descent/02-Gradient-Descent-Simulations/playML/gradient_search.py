def safe(f):
    def safe_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            return float('inf')

    return safe_f


def gradient_descent(J, dJ, initial_theta, eta,
                     n_iters=1e4, epsilon=1e-8):

    J = safe(J)
    theta = initial_theta
    iter = 1

    while iter <= n_iters:
        gradient = dJ(theta)
        last_theta = theta
        theta = theta - eta * gradient

        if (abs(J(theta) - J(last_theta)) < epsilon):
            break

        iter += 1

    return