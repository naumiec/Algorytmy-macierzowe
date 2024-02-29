import pandas as pd
import matplotlib.pyplot as plt

def plot_time_error():
    non_zero = [0.01, 0.02, 0.05, 0.1, 0.2]
    b = [1, 1, 1, 4, 4, 4]
    sigma = ['$sigma_1$', '$sigma_{2^{k-1}}$', '$sigma_{2^{k}}$']
    time = [0.021960, 0.577161, 20.231883, 0.028359, 0.717139, 6.833700,
             0.023493, 0.544299, 29.033727, 0.025757, 0.258863, 8.499308,
             0.022849, 0.221370, 64.627048, 0.023469, 0.238199, 15.988675,
             0.021120, 0.217040, 114.064481, 0.024118, 0.257509, 26.809606,
             0.020499, 0.198258, 199.693175, 0.022919, 0.234090, 48.766644]
    error = [5.875743e+01, 5.102409e+01, 8.500174e-15,
              5.838015e+01, 4.558788e+01, 4.074786e-14,
              8.286845e+01, 7.606209e+01, 2.445790e-14,
              8.237188e+01, 7.485865e+01, 6.550157e-14,
              1.298218e+02, 1.264752e+02, 2.026586e-14,
              1.290964e+02, 1.200415e+02, 1.131958e-13,
              1.792990e+02, 1.752112e+02, 2.877998e-14,
              1.783093e+02, 1.669733e+02, 2.318141e-03,
              2.430412e+02, 2.379345e+02, 4.155036e-14,
              2.416912e+02, 2.273164e+02, 8.446529e-04]

    # Plot for time
    plt.subplot(1, 2, 1)
    plt.plot(non_zero, time[::3], label=sigma[0])
    plt.plot(non_zero, time[1::3], label=sigma[1])
    plt.plot(non_zero, time[2::3], label=sigma[2])
    plt.xlabel('Non-zero values')
    plt.ylabel('Time')
    plt.title('Time for Different Sigma and b')
    plt.legend()

    # Plot for error
    plt.subplot(1, 2, 2)
    plt.plot(non_zero, error[::3], label=sigma[0])
    plt.plot(non_zero, error[1::3], label=sigma[1])
    plt.plot(non_zero, error[2::3], label=sigma[2])
    plt.xlabel('Non-zero values')
    plt.ylabel('Error')
    plt.title('Error for Different Sigma and b')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_time_error()
