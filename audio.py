import pyworld as pw
import pysptk as sptk
import sounddevice as sd
from scipy.io import wavfile
import numpy as np
def debug_synth(f0,sp,ap,fs,an=2):
    """
    Plays a synthetised audio

    Parameters:
    -----------
    Parameters from the PyWORLD vocoder

    f0: fundamental frequency
    sp: spectrum
    ap: band aperiodicities
    fs: sampling frequency (typically 16000)
    an: length of the analysis window

    """
    
    sound = pw.synthesize(f0,sp,ap,fs, an)
    sd.play(sound,fs)
    sd.wait()

def debug_resynth(f0_,sp_,ap_,fs,an=2,alpha=0.42,fftbin=1024):
    """
    Plays a synthetised audio from the encoded parameters
    It is good to perform quick analysis-resynthesis

    Parameters:
    -----------
    Parameters from the PyWORLD vocoder

    f0: fundamental frequency
    sp: spectrum
    ap: band aperiodcities
    fs: sampling frequency (typically 16000)
    an: length of the analysis window
    alpha: pre-emphasis filtering coefficient
    fftbin: bin-size of the underlying FFT 
    """
    sp_ = sptk.conversion.mc2sp(sp_, alpha, fftbin)
    ap_ = pw.decode_aperiodicity(ap_, fs, fftbin)
    sound = pw.synthesize(f0_,sp_,ap_,fs,an)
    sd.play(sound,fs)
    sd.wait()

    return sound
def save_resynth(fname,f0_,sp_,ap_,fs,dtype=np.float32,an=2,alpha=0.42,fftbin=1024):
    """
    Plays a synthetised audio from the encoded parameters
    It is good to perform quick analysis-resynthesis
    It also saves the file in WAV format

    Parameters:
    -----------
    Parameters from the PyWORLD vocoder

    f0: fundamental frequency
    sp: spectrum
    ap: band aperiodcities
    fs: sampling frequency (typically 16000)
    an: length of the analysis window
    alpha: pre-emphasis filtering coefficient
    fftbin: bin-size of the underlying FFT 

    """
    sp_ = sptk.conversion.mc2sp(sp_, alpha, fftbin)
    ap_ = pw.decode_aperiodicity(ap_, fs, fftbin)
    sound = pw.synthesize(f0_,sp_,ap_,fs,an)
#    sd.play(sound*3,fs)
#    sd.wait()
    wavfile.write(fname,fs,sound.astype(dtype))
    return sound
