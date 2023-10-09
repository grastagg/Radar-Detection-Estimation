import numpy as np
import scipy
import matplotlib.pyplot as plt



def signal_to_noise_ratio(P_t, tau, G_t, G_r, radar_wavelength, radar_cross_section, T_s, dist, loss):
    num = np.float128((P_t * tau*G_t*G_r*(radar_wavelength**2)*radar_cross_section))
    den = np.float128((4*np.pi)**3*scipy.constants.Boltzmann*T_s*dist**4*loss)
    return num/den

def frequency_to_wavelength(frequency):
    return 299792458.00/frequency

def detectability(probablility_of_detection, probability_of_false_alarm, swirling_model):
    if swirling_model == 0:
        return 0
    elif swirling_model == 1 or swirling_model == 2:
        return (np.log(probability_of_false_alarm))/(np.log(probablility_of_detection))-1
    elif swirling_model == 3 or swirling_model == 4:
        return 
    

def main():
    lamb = frequency_to_wavelength(3e9) # Wavelength (m)
    Pt = 0.2e6                          # Peak power (W)
    tau = 1.1e-5                        # Pulse width (s)
    Gt = 10**(34/10)                             # Transmit and receive antenna gain (dB)
    # print(Gt)
    Ts = 745.4148                            # System temperature (K) 
    rcs = 1                             # Target radar cross section (m^2)
    
    
    r = np.linspace(1,130e3,1001)
    
    # snr = 10*np.log10(signal_to_noise_ratio(Pt, tau, Gt, Gt, lamb, rcs, Ts, r, 1))
    # print(snr[0])
    # # print(snr)
    # plt.figure()
    # plt.plot(r, snr)
    # plt.show()
    Pd = 0.9
    Pfa = 1e-6
    detect = detectability(Pd, Pfa)
    print(10*np.log10(detect))
    

if __name__ == '__main__':
    main()