##Pitch fixed- Rahil 

import sys
import numpy as np
import scipy.io as si
import scipy.signal as ss
#import sounddevice as sd
import matplotlib.pyplot as plt
import matplotlib.image as image

def nextpow2(i):
    n = 1
    while n < i: n *= 2
    return n


def unitseq ( x ):
    x = x - np.mean(x)
    x = x / np.std(x,ddof=1)
    return x

    
def sigmoid ( y, fac):
    """
# SIGMOID nonlinear funcion for cochlear model
# y = sigmoid(y, fac);
#    fac: non-linear factor
#     -- fac > 0, transistor-like function
#     -- fac = 0, hard-limiter
#     -- fac = -1, half-wave rectifier
#     -- else, no operation, i.e., linear 
#
#    SIGMOID is a monotonic increasing function which simulates 
#    hair cell nonlinearity. 
#    See also: WAV2AUD, AUD2WAV
#
# Author: Powen Ru (powen@isr.umd.edu), NSL, UMD, python version by Nina Burns
# v1.00: 01-Jun-97
# vpython: 10-12-2016
"""
    
    if fac > 0:
        y = np.exp(-y/fac)
        y = 1.0/(1+y)
    elif fac == 0:
        y = np.where(y>0,1,0) # step
    elif fac == -1:
        y = np.max(y,0)
    #else if fac == -3:
    #    y = halfregu(y)
        
    return y
        

def wav2aud( x, paras, filt='p', verbose=False ):

    """
    function v5 = wav2aud(x, paras, filt, verbose)
% WAV2AUD fast auditory spectrogramm (for band 180 - 7246 Hz)
%    v5 = wav2aud(x, [frmlen, tc, fac, shft], filt, VERB);
%    x    : the acoustic input.
%    v5    : the auditory spectrogram, N-by-(M-1) 
%
%    COCHBA  = (global) [cochead; cochfil]; (IIR filter)
%   cochead : 1-by-M filter length (<= L) vector.
%               f  = real(cochead); filter order
%               CF = imag(cochead); characteristic frequency
%    cochfil : (Pmax+2)-by-M (L-by-M) [M]-channel filterbank matrix.
%        B = real(cochfil); MA (Moving Average) coefficients.
%        A = imag(cochfil); AR (AutoRegressive) coefficients.
%    M    : highest (frequency) channel 
%
%    COCHBA  = [cochfil]; (IIR filter)
%    cochfil : (L-by-M) [M]-channel filterbank impulse responses.
%
%    PARAS    = [frmlen, tc, fac, shft];
%    frmlen    : frame length, typically, 8, 16 or 2^[natural #] ms.
%    tc    : time const., typically, 4, 16, or 64 ms, etc.
%          if tc == 0, the leaky integration turns to short-term avg.
%    fac    : nonlinear factor (critical level ratio), typically, .1 for
%          a unit sequence, e.g., X -- N(0, 1);
%          The less the value, the more the compression.
%          fac = 0,  y = (x > 0),   full compression, booleaner.
%          fac = -1, y = max(x, 0), half-wave rectifier
%          fac = -2, y = x,         linear function
%    shft    : shifted by # of octave, e.g., 0 for 16k, -1 for 8k,
%          etc. SF = 16K * 2^[shft].%    
%
%    filt    : filter type, 'p'--> Powen's IIR filter (default)
%                   'p_o' --> Powen's old IIR filter (steeper group delay)    
%    
%    IIR filter : (24 channels/oct)
%    for the output of     downsamp/shift    tc (64 ms)/ frame (16 ms)
%    ==================================================================
%    180 - 7246        1    /0    1024    / 256
%    90  - 3623        2    /-1    512    / 128    *
%
%    Characteristic Frequency: CF = 440 * 2 .^ ((-31:97)/24);
%    Roughly, CF(60) = 1 (.5) kHz for 16 (8) kHz.
%
%    verbose    : verbose mode
%
%    WAV2AUD computes the auditory spectrogram for an acoustic waveform.
%    This function takes the advantage of IIR filter's fast performance
%    which not only reduces the computation but also saves remarkable
%    memory space.
%    See also: AUD2WAV, UNITSEQ

% Author: Powen Ru (powen@isr.umd.edu), NSL, UMD
% v1.00: 01-Jun-97
# Python version:  Nina K. Burns, MITRE, Inc.
# v1.00: October 13, 2016
"""


    # get filter bank,
    #    L: filter coefficient length;
    #    M: no. of channels
    
    if (filt=='k'):
       print('Please use wav2aud_fir function for FIR filtering!')
       return
    
    if (verbose == True):
        octband = None
    
    if (filt == 'p_o'):
        COCHBA = np.array(list(si.loadmat('data/aud24_old.mat').values()))[0]
    else:
        COCHBA = np.array(list(si.loadmat('data/aud24.mat').values()))[0]
    
    [L, M] = np.shape(COCHBA) #  p_max = L - 2
    L_x = len(x) # length of input
    
    #octave shift, nonlinear factor, frame length, leaky integration
    shft    = paras[3]   # octave shift
    fac    = paras[2]    # nonlinear factor
    L_frm = int(np.round(paras[0] * np.power(2,(4+shft))))   #frame length (points)
    
    if paras[1]:
        alph    = np.exp(-1/(paras[1]*np.power(2,(4+shft)))) # decay factor
    else:
        alph    = 0                    # short-term avg.
    
    #hair cell time constant in ms
    haircell_tc = 0.5
    beta = np.exp(-1/(haircell_tc*np.power(2,(4+shft))))
    
    #get data, allocate memory for ouput 
    N = int(np.ceil(L_x / L_frm))   # number of frames
    xlen = int(N*L_frm)
    x = np.pad(x,(0,xlen-len(x)),'constant')  # zero-padding
    v5 = np.zeros((N, M-1))
    # %CF = 440 * 2 .^ ((-31:97)/24);
    
    ####################################
    # last channel (highest frequency)
    ####################################
    p    = int(np.real(COCHBA[0,M-1]))
    B    = np.real(COCHBA[1:p+2, M-1])
    A    = np.imag(COCHBA[1:p+2, M-1]) 
    y1    = ss.lfilter(B, A, x) 
    y2    = sigmoid(y1, fac)
    
    # hair cell membrane (low-pass <= 4 kHz); ignored for LINEAR ionic channels
    if (fac != -2):
        y2 = ss.lfilter([1], [1,-beta], y2)
    
    y2_h = y2
    y3_h = 0
    
    #t0 = clock
    
    
    ####################################
    # All other channelsy2
    ####################################
    for ch in range((M-2),-1,-1):
        
        ####################################
        # ANALYSIS: cochlear filterbank
        ####################################
        # (IIR) filter bank convolution ---> y1
        p  = int(np.real(COCHBA[0, ch]))         # order of ARMA filter
        B  = np.real(COCHBA[1:p+2, ch])     # moving average coefficients
        A  = np.imag(COCHBA[1:p+2, ch])   # autoregressive coefficients
        y1 = ss.lfilter(B, A, x) 
        ####################################
        #TRANSDUCTION: hair cells
        ####################################
        # Fluid cillia coupling (preemphasis) (ignored)
    
        # ionic channels (sigmoid function)
        y2 = sigmoid(y1, fac);
    
        # hair cell membrane (low-pass <= 4 kHz) ---> y2 (ignored for linear)
        if (fac != -2):
            y2 = ss.lfilter([1], [1,-beta], y2)
        
        #################################### 
        # REDUCTION: lateral inhibitory network
        ####################################
        # masked by higher (frequency) spatial response
        y3   = y2 - y2_h
        y2_h = y2
    
        # spatial smoother ---> y3 (ignored)
        #y3s = y3 + y3_h;
        #y3_h = y3;
    
        # half-wave rectifier ---> y4
        y4 = np.maximum(y3, np.zeros_like(y3))
    
        # temporal integration window ---> y5
        if (alph):    # leaky integration
            y5 = ss.lfilter([1], [1,-alph], y4)
            v5[:,ch] = y5[(L_frm*np.arange(1,N+1))-1]
        else:        # short-term average
            if (L_frm == 1):
                v5[:, ch] = y4
            else:
                v5[:, ch] = np.mean(y4.reshape((L_frm, N))).transpose() # watch order
    
        if (verbose == True and filt == 'p'):
            if np.remainder(ch, 24) == 0:
                if octband == None:
                    octband = octband + 1
                else:
                    octband = 1
            print('%d octave(s) processed\r' % octband)
    
    if verbose is True:
        print('\n')
        
    return v5

  
def aud2cors(y, sv, SRF=24, FULL=0, BP=0):

# AUD2CORS static cortical representation.
#	z = aud2cors(y, sv);
#	z = aud2cors(y, sv, SRF, FULL, BP);
#	y	: auditory spectrum (M-by-1)
#	sv	: char. ripple freq's (K-by-1), e.g., sv = 2.^(-2:.5:3)
#	SRF	: (optional) sample ripple frequency, default = 24 ch/oct
#	z	: cortical representation (M-by-K)
#	FULL: non-truncation factor
#	BP	: all bandpass filter bank
#
#	AUD2CORS computes the K-channel cortical representation for an 
#	auditory spectrum Y. The ANGLE(Z) represents the symmetry the unit
#	with the maximum response ABS(Z). For display purpose, the complex
#	matrix Z should be encoded to the indice of a specific colomap by
#	an utility program CPLX_COL.
#	See also: WAV2AUD, AUD2CORS, CPLX_COL, COR_DIST, COR_MAPS
#
#   Author: Powen Ru (powen@isr.umd.edu), NSL, UMD
#   Python version:  Nina Burns, MITRE Inc. 

    #y = y[:]
    K = len(sv)
    M = len(y)
    M1 = nextpow2(M)
    M2 = M1 * 2
    Mp = np.round((M2-M)/2)

    # 24 channel per octave default
    # FULL=0 default with truncation
    # BP = 0 all bandpass filter bank

    # fourier transform
    ypad = np.arange(1,M2+1-M)/(M2-M+1) * (y[0] - y[M-1])
    ypad = y[M-1]+ypad

    y = np.concatenate((y,ypad))
    Y = np.fft.fft(y)
    Y = Y[:M1]

    # spatial filtering
    dM = np.floor(M/2*FULL)
    if FULL > 0:        # what is truncation factor? Fix indices for zero start
        one=np.arange(1,dM)+M2-dM
        two=np.arange(1,M+dM)
        mdx = np.concatenate(np.arange(1,dM)+M2-dM,np.arange(1,M+dM))
    else:
        mdx = np.arange(0,M)

    z = np.zeros((M+int(2*dM), K),dtype=np.complex128)

    for k in range(0,K):
        H = gen_corf(sv[k], M1, SRF, np.array([k+1+BP, K+BP*2]))
        R1 = np.fft.ifft(H*Y, M2, axis=0)
        z[:, k] = R1[mdx]

    return z


def gen_corf(fc, L, SRF, KIND=2):
#% GEN_CORF generate (bandpass) cortical filter transfer function
#%	h = gen_corf(fc, L, SRF);
#%	h = gen_corf(fc, L, SRF, KIND);
#%	fc: characteristic frequency
#%	L: length of the filter, power of 2 is preferable.
#%	SRF: sample rate.
#%	KIND: (scalar)
#%	      1 = Gabor function; (optional)
#%	      2 = Gaussian Function (Negative Second Derivative) (default)
#%	      (vector) [idx K]
#%	      idx = 1, lowpass; 1<idx<K, bandpass; idx = K, highpass.
#%
#%	GEN_CORF generate (bandpass) cortical filter for various length and
#%	sampling rate. The primary purpose is to generate 2, 4, 8, 16, 32 Hz
#%	bandpass filter at sample rate 1000, 500, 250, 125 Hz. This can also
#%	be used to generate bandpass spatial filter .25, .5, 1, 2, 4 cyc/oct
#%	at sample ripple 20 or 24 ch/oct. Note the filter is complex and
#%	non-causal.
#%	see also: AUD2COR, COR2AUD
 
#% Author: Powen Ru (powen@isr.umd.edu), NSL, UMD
# Python version:  Nina K. Burns, MITRE

    if len(KIND) == 1:
        PASS = np.array([2, 3])	# bandpass
    else:
        PASS = KIND
        KIND = 2

# fourier transform of lateral inhibitory function 

# tonotopic axis
    # watch order
    R1	= np.reshape(np.arange(L),(128,)).T/L*SRF/2/np.abs(fc)	# length = L + 1 for now

    if KIND == 1:	# Gabor function
        C1 = ((1/2)/.3)/.3
        H = np.exp(-C1*(R1-1)**2) + np.exp(-C1*(R1+1)**2)
    else:		# Gaussian Function 
        R1 = R1**2
        H = R1 * np.exp(1-R1) 	# single-side filter

#H = H * (1-exp(-(fc/.5).^0.5))

# passband
    if PASS[0] == 1:		#lowpass 
        maxH = np.amax(H)
        maxi = np.argmax(H)
        sumH = np.sum(H)
        H[0:maxi] = np.ones((maxi,))
        H = H / np.sum(H) * sumH
    elif PASS[0] == PASS[1]:	# highpass check this
        # maxH = np.amax(H)
        maxi = np.argmax(H)
        sumH = np.sum(H)
        H[maxi:L] = np.ones((L-maxi,))
        H = H / np.sum(H) * sumH

    return H

def pitch(S,f,fname,npeak=1):

    """
    # [pit,sal,pitches,c] = pitch(S,f,fname)
    #
    # pitch: compute pitch from auditory spectrogram 
    #
    # INPUTS
    # S: auditory spectrogram
    # f: freqencies corresponding to the rows of S
    # fname: file name where pitch templates are kept
    #
    # OUTPUTS
    # pit: pitch estimate
    # sal: saliency of pitch estimate
    # pitches: matrix of correlation peaks and values
    # c: cross-correlation functions
    """
    
    [m,n] = np.shape(S)  # m = number of channels,  n = number of time samples
    
    c = np.zeros((m,n)) # template crosscorrelation result
    #pad=49
    # Loads the harmonic templates (ts) plus hs and pad?
    ts = si.loadmat(fname+'.mat')['ts']
    ts = np.reshape(ts,(177,)) # watch order
    tmp = si.loadmat(fname+'.mat')['pad']
    # print ('pitch tmp=',tmp)
    #tmp=[[49]]
    # pad = tmp[0,0]
    pad=130 #hard-coded padding params. Need to automate this 
    # print (len(ts))
    # print ('pad=', pad)
    # print ('m=',m)
    # print (pad)

    # dim1: xcorr length, dim2: time samples, may have to re-think padding, etc.
    tslen = np.shape(ts)[0]
    # # plt.imshow(ts)
    # plt.plot(ts)
    # plt.show()
    dim1 = 2*max(tslen,m)   
    temp = np.zeros((dim1-1,n))
    for j in np.arange(0,n): # for each time
        temp[:,j] = np.flipud(np.correlate(ts,np.pad(S[:,j],(0,tslen-m),'constant'),'full'))
    
    # recenter to pitch coordinates
    c = temp[(tslen-pad):(tslen-pad+m),:] #this is the line for testing params

    # c = temp[(tslen-pad):(tslen-pad+m),:] #this is the OG line
    # print ('temp=',np.shape(temp))
    # print ('c=', np.shape(c))
    # print ('tslen-pad', tslen-pad)
    # print ('tslen-pad+m', tslen-pad+m)
    # print ('tslen-', tslen)
    # image.imsave('./temp_pitch.png',temp,cmap='jet',origin='lower')
    # image.imsave('./c_pitch.png',c,cmap='jet',origin='lower')

    npeak = 4  # number of simultaneous pitches to look for
    th = 0.5   # A threshold factor
    pitches = np.zeros((m,n))
    
    for tx in range(0,n):
    
        slice = c[:,tx]
    
        # Relative threshold -- masking effect
        slice = np.maximum(slice-th*np.amax(slice),0);
     
        # Peaks are then selected within each channel
        [ch,peak] = pickpeaks(slice,npeak)
     
        # Sharpening and normalization
        if np.sum(slice):
            salmap = peak**2/(np.sum(slice))
        else:
            salmap = peak**2
    
        # Here the npeak highest peaks are selected, check: might be ch >=0
        pitches[ch[np.where(ch>=0)],tx] = salmap[np.where(ch>=0)]
    
    # This is the highest peak - the dominant pitch
    # check: here I don't understand the matlab code...max over 3rd dimension?

    sal = np.amax(pitches,axis=0)
    pit = np.argmax(pitches,axis=0)
    # print ('pit-', pit)
    # print ('np.shape(pit)', np.shape(pit)).  #pit is a 989 long np vector
    # print (len(f)) #len of f =127
    # print ('np.minimum(pit+1,len(f))', np.minimum(pit+1,len(f)))
    temp_var=np.minimum(pit,len(f))
    # print (temp_var)
    # print (f)
    # print (np.shape(f))
    pit = f[np.minimum(pit,len(f)-1)] # modified on April 17, 2019-Rahil

    # pit = f[np.minimum(pit+1,len(f))] # hack... this should be taken care of by ts (check: what is this?)
    pit = pit[:]
    sal = sal[:]
    
    return (pit,sal,pitches,c)


#def permute():
#
#return


def cochfil(n, shft=0):

    # COCHFIL cochlear filter coefficient reader
    #	[CF, B, A] = cochfil(n, shft);
    #	HH = cochfil(CHAR);
    #	n: channel indices, e.g., 60 or 11:100
    #	shft: octave shift
    #	CF: characteristic frequencies of the selected channels
    #	B, A: IIR coefficients of the selected channel (only one)
    #	HH: CHAR = 'o', overall response;
    #		CHAR = 'n', normalized overall response.
    
    # Author: Powen Ru Neural Systems Laboratory, Univ. of MD College Park
    # Python version:  Nina K. Burns, MITRE Inc.

    if (n[0] == 'o') or (n[0] == 'n'):	# overall response,
        [N, M] = np.shape(COCHBA)
        CF = 440 * 2**((np.arange(1,M+1)-31)/24)/16000
        s = np.exp(2*1j*np.pi*CF)
        HH2 = 0
        for m in range(0,M):
            p = np.real(COCHBA[0, m])
            B = np.real(COCHBA[1:p+1, m])
            A = np.imag(COCHBA[1:p+1, m])
            h = np.divide(np.polyval(B,s),np.polyval(A,s))
            if n[0] == 'o':
                HH2 = HH2 + np.abs(h)**2
            else:
                NORM = np.imag(COCHBA[0,m])
                HH2 = HH2 + np.abs(h)**2 / NORM
        CF = HH2
        B = []
        A = []

    else:
	# filterbank
        if len(n) > 1:	# no coefficient output
            B = []
            A = []
        else:			# coefficient output
            if (n > 0) and (n < np.size(COCHBA, 1)):
                p = np.real(COCHBA[1, n])
                B = np.real(COCHBA[2:p+2, n])
                A = np.imag(COCHBA[2:p+2, n])
            else:
                print('invalid channel index!')	

	# characteristic frequency
        CF = 440 * 2**((n-31)/24+shft)

    return CF

def pickpeaks(signal,num=2,rdiff=5):

#% [loc,val] = pickpeak(spec,npicks,rdiff)
#%       spec   - data vector 
#%       npicks - number of peaks desired              [default = 2]
#%       rdiff  - minimum spacing between picked peaks [default = 5]
#%       loc    - vector of locations (indices) of the picked peaks
#%       val    - vector corresponding values 
#%       A 0 in location (i,j) of array loc (or a NaN in array val)
#%       indicates that the j-th data vector has less than i peaks
#%       with a separation of rdiff or more. 

    (m,)  = np.shape(signal)
    if m == 1:
        signal = np.transpose(signal)
        [m,n] = np.shape(signal)

    if np.invert(np.isfinite(signal)).any() is True:
        sys.exit("Found NaN")
    
    rmin = np.amin(signal) - 1

    # find a peak for comparison to other peaks

    val =  np.ones((num,))*np.nan
    loc =  np.zeros((num,),dtype=int)

        
    x = np.pad(signal,(1,1),'constant',constant_values=rmin)
    dx = np.diff(x)
    vp = signal[np.all([dx[0:m]>=0,dx[1:m+1]<=0],axis=0)]
    lp = np.where(np.logical_and((dx[0:m]>=0),(dx[1:m+1] <=0)))[0]
    #vp = signal(lp,k)

    for p in np.arange(0,num):
       v = np.amax(vp)                # find current maximum
       l = np.argmax(vp)
       val[p] = v
       loc[p] = lp[l]		      # save value and location 
       
       # find peaks outside of rdiff
       ind = np.nonzero(np.abs(lp[l]-lp) > rdiff)

       if (np.all(ind==0)): 
           break                      # if no more local peaks

       vp  = vp[ind]                  # eliminate peak values
       lp  = lp[ind]                  # eliminate peak locations

    return [loc,val]


def aud2tf(y, rv, sv, STF, SRF=24, BP=1):
# AUD2TF Temporal or spatial filtering of an auditory spectrogram
#   [rtf, stf] = aud2tf(y, rv, sv, STF, SRF, BP);
#	y   : auditory spectrogram, N-by-M, where
#		N = # of samples, M = # of channels
#	stf : scale-time-frequency matrix, S-by-N-by-M, where 
#		S = # of scale
#	rtf : rate(up,down)-time-frequency matrix, 2R-by-N-by-M, where 
#		R = # of rate
#	rv  : rate vector in Hz, e.g., 2.^(1:.5:5).
#	sv  : scale vector in cyc/oct, e.g., 2.^(-2:.5:3).
#	STF	: sample temporal frequency, e.g., 125 Hz for 8 ms
#	SRF	: sample ripple frequency, e.g., 24 ch/oct or 20 ch/oct
#	BP	: pure bandpass indicator, default : 1
#
#   AUD2TF generates various spectrograms out of different temporal or
#	spatial filters with respect to the auditory spectrogram Y
#	which was generated by WAV2AUD. 
#	RV (SV) is the characteristic frequency vector.
#	See also: WAV2AUD, RST_VIEW, COR2RST

# Author: Taishih Chi (tschi@isr.umd.edu), NSL, UMD
# Python version:  Nina K. Burns, MITRE

    # dimensions
    K1 = len(rv)	 # of rate channel
    K2 = len(sv)	 # of scale channel
    [N, M] = np.shape(y)	# dimensions of auditory spectrogram

    # spatial, temporal zeros padding 
    N1 = nextpow2(N)
    N2 = N1*2
    M1 = nextpow2(M)
    M2 = M1*2

    # calculate stf (perform aud2cors frame-by-frame) 
    if K2 > 0:
        stf = np.zeros((2*K1, N, M))		# memory allocation
        for i in np.arange(1,N):
            stf[:, i, :] = np.conj(aud2cors(y[i, :], sv, SRF, 0, BP).transpose())
    else:
        stf = y

    # calculate rtf (perform filtering channel-by-channel)
    # compute rate filters
    HR = np.zeros((2*N1, 2*K1),dtype=np.complex128)
    for k in np.arange(0,K1):
        Hr = gen_cort(rv[k], N1, STF, np.array([k+1+BP, K1+BP*2]))
        Hr = np.concatenate((Hr, np.zeros((N1,))),axis=0)	# SSB -> DSB
        HR[:, k+K1] = Hr		# downward
        HR[:, k] = np.concatenate((np.array((Hr[0],)), np.conj(np.flipud(Hr[1:N2]))),axis=0)	# upward
        HR[N1, k] = np.abs(HR[N1+1, k])

    if K1 > 0:
        rtf = np.zeros((2*K1, N, M),dtype=np.complex128)		# memory allocation
        for i in np.arange(0,M):
            ypad = y[N-1, i] + (np.arange(1,(N2-N+1))/(N2-N+1) * (y[0, i] - y[N-1, i]))
            ytmp = np.concatenate((y[:, i], ypad),axis=0) # compatible dimensions?
            YTMP = np.fft.fft(ytmp, N2) # compatible dimensions?
            # temporal filtering
            for k in np.arange(0, 2*K1):
                R1 = np.fft.ifft(np.multiply(HR[:, k],YTMP), N2)
                rtf[k, :, i] = R1[0:N]
    else:
        rtf = y

    return rtf, stf



def gen_cort(fc, L, STF, PASS=[2,3]):
# GEN_CORT generate (bandpass) cortical temporal filter transfer function
#	h = gen_cort(fc, L, STF);
#	h = gen_cort(fc, L, STF, PASS);
#	fc: characteristic frequency
#	L: length of the filter, power of 2 is preferable.
#	STF: sample rate.
#	PASS: (vector) [idx K]
#	      idx = 1, lowpass; 1<idx<K, bandpass; idx = K, highpass.
#
#	GEN_CORT generate (bandpass) cortical temporal filter for various
#	length and sampling rate. The primary purpose is to generate 2, 4,
#	8, 16, 32 Hz bandpass filter at sample rate ranges from, roughly
#	speaking, 65 -- 1000 Hz. Note the filter is complex and non-causal.
#	see also: AUD2COR, COR2AUD, MINPHASE
 
# Author: Powen Ru (powen@isr.umd.edu), NSL, UMD
# Python version:  Nina K. Burns, MITRE


    # Gamma distribution function 
    t = np.arange(0,L).transpose()/STF * fc
    h = np.multiply(np.sin(2*np.pi*t),t**2) * np.exp(-3.5*t) * fc

    h = h-np.mean(h)
    H0 = np.fft.fft(h, 2*L)
    A = np.angle(H0[0:L])
    H = np.abs(H0[0:L])
    maxH  = np.amax(H)
    maxi = np.argmax(H)
    H = H / np.amax(H)

    # passband
    if PASS[0] == 1:		# lowpass
        H[0:maxi] = np.ones(maxi,) 
    elif PASS[0] == PASS[1]:	# highpass
        H[maxi:L] = np.ones((L-maxi, )) # check

    H = np.multiply(H, np.exp(1j*A))

    return H

def aud_fix(y):

    # AUD_FIX fix the auditory spectrogram
    # 	y = aud_fix(y);
    #	y (i/p): complex matrix
    #	y (o/p): non-negative real matrix
    #	
    #	AUD_FIX fixes the auditory spectrogram by taking the real part
    #	of the input, then half-wave rectifying it. Some more processes
    #	may be added in order to improve the sound quality. This 
    #	function is employed to fix the reconstructed auditory 
    #	spectrogram generated by COR2AUD or COR2AUDS.
    #	See also: COR2AUD, COR2AUDS, AUD_MAPS

    # Author: Powen Ru (powen@isr.umd.edu), NSL, UMD
    # Python version:  Nina K. Burns, MITRE

    y = np.real(y)
    y = np.maximum(y, 0)

    return y


def aud2wav(v5, x0, paras):

    # AUD2WAV fast inverse auditory spectrum (for band 180 - 7246 Hz)
    #	[x0, xmin, errv] = aud2wav(v5, x0, [L_frm, tc, fac, shft, ...
    #			iter, DISP, SND])	
    #	v5		: auditory spectrogram (N-by-M)
    #	x0		: the projected (guessed) acoustic output (input).
    #	xmin	: the sequence with minimum error
    #	errv	: error vector.
    #
    #	COCHBA  = (global) [cochead, cochfil] 
    #		cochead : 1-by-M, M-channel header
    #		p  = real(cochead) filter order
    #		CF = imag(cochead) characteristic frequency
    #	cochfil : (L-1)-by-M, M-channel filterbank
    #		B = real(cochfil) MA (Moving Average) coefficients.
    #		A = imag(cochfil) AR (AutoRegressive) coefficients.
    #
    #	PARAS	= [L_frm, tc, fac, shft, iter, DISP, SND]
    #	L_frm	: frame length, typically, 16 ms or 2^[natural #] ms.
    #	tc	: time const, typically, 64 ms = 1024 pts for 16 kHz.
    #		if tc == 0, the leaky integration turns to short-term average
    #	fac	: nonlinear factor (critical level ratio), typically, .01.
    #		The less the value, the more the compression
    #			fac = 0: y = (x > 0), full compression
    #			fac = -1, y = max(x, 0), half-wave rectifier
    #			fac = -2, y = x, linear function
    #	shft	: shifted by # of octave, e.g., 0 for 16k, -1 for 8k,
    #			etc. SF = 16K * 2^[shft].
    #	iter	: # of iterartions
    #	DISP	: display the new spectrogram (1) or not (0).
    #	SND	: play the sound (1) or not (0).
    #
    #	AUD2WAV inverts auditory spectrogram V5 back to acoustic input.
    #	The COCHBA (in AUD24.MAT) should have been loaded and set to 
    #	global beforehand.
    #	See also: WAV2AUD	

    # Author: Powen Ru (powen@isr.umd.edu), NSL, UMD
    # Revision: Taishih Chi (tschi@isr.umd.edu), NSL, UMD
    # Python version:  Nina K. Burns, MITRE

    # get filter bank,
    #	L: max. # of order + 2;
    #	M: no. of channels

    COCHBA = np.array(list(si.loadmat('data/aud24.mat').values()))[0]
    [L, M] = np.shape(COCHBA)	# p_max = L - 2

    # options
    INIT = 1		# initial channel (highest freq.)
    DIFF = 0		# difference matching
    DEOVR  = 0		# de-overlapping

    # extract parameters 
    shft = paras[3]	# octave shift
    fac	= paras[2]	# nonlinear factor
    # make sure this is int?
    L_frm = int(paras[0] * 2**(4+shft))	# frame length (in points)

    if paras[1]:
        alph = np.exp(-1/(paras[1]*2**(4+shft)))# leaky integ. coeff.
        alp1 = np.exp(-paras[1]/paras[1])	# for one frame
    else:
        alph = 0				# short-term average
        alp1 = 0

    haircell_tc=0.5				# hair cell time constant
    beta = np.exp(-1/(haircell_tc*2**(4+shft)))

    iter = paras[4]				# of iter
    DISP = paras[5]				# image/plot
    SND	= paras[6]				# play sound for each iter.

    # fix the gcf
    if DISP:
      fig_aud  = gcf

    # fix auditory spectrum
    if np.isreal(v5) is False:
        v5 = aud_fix(v5)

    # get data, allocate memory for ouput 
    import copy
    [N, MM] = np.shape(v5)
    v5_new = copy.deepcopy(v5)
    v5_mean = np.mean(v5)
    v5_sum2 = np.sum(v5**2)
    L_x	= int(N * L_frm)

    # de-overlapping vector
    vt = copy.deepcopy(v5)	# target vectors
    if DEOVR:
        for n in np.arange(1,N):
            vt[n,:] = np.maximum((vt[n,:]-alp1*vt[n-1,:],0))

    # initial guess
    L_x0 = int(len(x0))
    x0 = x0.flatten() # watch order
    if L_x0 == 0:
        x0 = np.random.rand(L_x)-.5	# uniform random sequence
        x0 = unitseq(x0)		# N(0, 1)
    elif L_x0 < L_x:
        padlen=L_x-L_x0
        x0=np.pad(x0,(0,padlen),mode='constant',constant_values=(0)) # zero-padding	
    else:
        x0 = x0[0:L_x]			# truncation

    # iteration
    xmin = x0	   # initial sequence with minimum-error
    emin = np.inf  # initial minimum error
    errv = []	   # error vector

    for idx in np.arange(0,iter):
            
        # normalization (Norm(0, 1))
        if fac == 0:	# default: No
            x0 = unitseq(x0)

        # projected v5
        ##########################################
        # last channel
        ##########################################

        if INIT:	# default: No
            p		= int(np.real(COCHBA[0, M-1]))
            NORM	= np.imag(COCHBA[0, M-1])
            B		= np.real(COCHBA[1:p+2, M-1])
            A		= np.imag(COCHBA[1:p+2, M-1])
            y1_h	= ss.lfilter(B, A, x0) 		# forward filtering
            y2_h	= sigmoid(y1_h, fac)		# nonlinear oper.

            if (fac != -2):
                y2_h = ss.lfilter([1], [1 -beta], y2_h)
            y_cum	= 0 # reverse filtering

        else:
            y1_h	= 0
            y2_h	= 0
            y_cum	= 0

        ##########################################
        # All other channels
        ##########################################

        for ch in np.arange(M-2,-1,-1):

            p = int(np.real(COCHBA[0, ch]))	# order of ARMA filter
            NORM =  np.imag(COCHBA[0, M-1])
            B	= np.real(COCHBA[1:p+2,ch])	# moving average coeff.
            A	= np.imag(COCHBA[1:p+2,ch])	# auto-regressive coeff.

            # forwarding
            y1 = ss.lfilter(B, A, x0)		# filter bank
            y2 = sigmoid(y1, fac)		# nonlinear op.
            if (fac != -2):
                y2 = ss.lfilter([1], [1,-beta], y2)

            y3 = y2 - y2_h			# difference (L-H)
            y4 = np.maximum(y3, 0)		# half-wave rect.
            if alph:
                y5 = ss.lfilter([1], [1,-alph], y4) # leaky integ. 
                vx = y5[np.arange(L_frm-1,N*L_frm,L_frm)]	# new aud. spec.
            else:
                vx = np.mean(y4.reshape(L_frm, N),axis=0) # watch order

            v5_new[:, ch] = vx

            # matching
            s = np.ones((N,1))
            for n in np.arange(0,N):
                if DEOVR and alp1 and (n>0):
                    vx[n] = np.maximum(vx[n]-alp1*vx[n-1], 0)

                # scaling vector
                if vx[n] > 0.0:
                    s[n] = vt[n, ch] / vx[n]
                elif vt[n, ch] > 0.0:
                    s[n] = 2	# double it for now
                else:
                    s[n] = 1
                    
                #?? hard scaling
            s = (s * np.ones((1,L_frm))).transpose()
            s = s.flatten('F')

            if (fac == -2):			# linear hair cell
                dy = y3
                y1 = np.multiply(dy,s)
            else:				# nonlinear hair cell
                ind = np.where(y3 >= 0)
                y1[ind] = y3[ind]*s[ind]
                maxy1p = np.amax(y1[ind])
                ind = np.where(y3 < 0)
                y1[ind] = y3[ind]/np.abs(np.amin(y3[ind]))*maxy1p

            y1_h = y1
            y2_h = y2

                # inverse wavelet transform
            y1 = ss.lfilter(B, A, np.flipud(y1))	# reverse filtering
            y_cum = y_cum + np.flipud(y1)/NORM	# accumulation

        # previous performance
        v5_r = v5_new / np.mean(np.mean(v5_new)) * v5_mean	# relative v5
        err = np.sum(np.sum((v5_r - v5)**2)) / v5_sum2	# relative error
        err = np.round(err * 10000) / 100
        era = np.sum(np.sum((v5_new - v5)**2)) / v5_sum2	# absolute error
        era = np.round(era * 10000) / 100	
        errv.extend([err,era])

        if err < emin:			# minimum error found
            emin = err
            xmin = x0
        elif (err-100) > emin:		# blow up !
            y_cum = unitseq(np.sign(y_cum)+np.random.rand(np.shape(y_cum)[0]))

        # inverse filtering/normalization
        x0 = y_cum*1.01		# pseudo-normalization

        # play sound (if necessary)
        if SND:
            if shft != -1:
                R1 = np.interp(np.round(np.arange(0,len(x0)*2**(shft+1)),
                    np.arange(0,len(x0)),x0))
            else:
                R1 = x0

            R1 = R1/np.amax(R1)*.9
            #sd.play(R1, 8000)
            #sd.stop()

        # display performance message
        print('{}, Err.: {} (rel.) {} (abs.) Energy: {}'.format(idx, err, era, 
            np.sum(x0**2)))

        # plot the auditory spectrogram (if necessary)
        if DISP:
            import matplotlib.pyplot as plt
            plt.imshow(v5_new,cmap='jet',interpolation='nearest')
            plt.title('Error')
            plt.show()


    print('Minimum Error: {} %'.format(emin))

    return x0, xmin, errv


def cor2audr(z, rv, STF, BP):

    # COR2AUDR complex inverse wavelet transform.
    #	tsf = cor2audr(z, rv)
    #	tsf = cor2audr(z, rv, STF)
    #	z: cortical representation, T-M*len(sv)-2*R, T = # of time frames
    #	rv: rate vector, R-by-1
    #	STF: (optional) sample  temporal frequency e.g., 125 Hz for 8 ms
    #	COR2AUDR reconstructs auditory spectrum at different scales from a complex
    #	cortical response Z with respect to rate vector rv.
    #   The default frame resolution is 8ms i.e 125 frames/sec.
    #	See also: AUD2CORS, COR2AUD , COR2AUD2

    # Author: Lakshmi Krishnan (lakshmik@umd.edu), NSL, UMD
    # Python version: Nina K. Burns, MITRE

    # check syntax
    [T,MS,R] = np.shape(z)
    tsf = np.zeros((T,MS),dtype=np.complex128)

    if (2*len(rv)) != R:
        print(error('size(z,3) ~= 2 * len(rv)'))
        sys.exit(-1)

    # normalization
    K1 = len(rv)
    N = T
    N1 = nextpow2(N)
    N2 = N1*2

    HR = np.zeros((2*N1, 2*K1),dtype=np.complex128)
    for k in np.arange(0,K1):
        Hr = gen_cort(rv[k], N1, STF, [k+BP, K1+BP*2])
        Hr = np.conj(np.concatenate((Hr, np.zeros_like(Hr))))# SSB -> DSB
        HR[:, k+K1] = Hr                # downward
        HR[:, k] = np.concatenate(([Hr[0]], np.conj(np.flipud(Hr[1:N2])))) #upward
        HR[N1, k] = np.abs(HR[N1+1, k])

    Z_cum = np.zeros((N2,MS),dtype=np.complex128)
    Hsum = np.zeros((N2,),dtype=np.complex128)

    # cumulation
    for r in np.arange(0,len(rv)):

        for sgn in [-1, 1]:

            # rate filtering modification
            if sgn < 0:

                for i in np.arange(0,np.size(z,1)):
                    ZF	= np.fft.fft(np.squeeze(z[:,i,r+len(rv)]),N2)
                    Z_cum[:,i] = Z_cum[:,i] + ZF*HR[:,r+K1]
                Hsum = Hsum +HR[:,r+K1]**2

            else:

                for i in np.arange(0,np.size(z,1)):
                    ZF	= np.fft.fft(np.squeeze(z[:,i,r]), N2)
                    Z_cum[:,i] = Z_cum[:,i] + ZF*HR[:,r]
                Hsum = Hsum +HR[:,r]**2

    NORM =0.99
    Hsum = Hsum*2
    sumH = np.sum(Hsum)
    Hsum = NORM * Hsum + (1 - NORM) * np.amax(Hsum)
    Hsum = Hsum / np.sum(Hsum) * sumH + 1	 # normalization for DC

    for i in np.arange(0,np.shape(z)[1]):
        Z_cum[:,i] = Z_cum[:,i] / Hsum
        tmp =  np.fft.ifft(Z_cum[:,i])
        tsf[:,i] = tmp[0:T]

    return tsf


def cor2auds(z, sv, NORM=.99, SRF=24, M0=128, FULLOUT=0, BP=0):

    # COR2AUDS complex inverse wavelet transform.
    #	yh = cor2auds(z, sv);
    #	yh = cor2auds(z, sv, NORM, SRF, M0, FULLOUT, BP);
    #	z: static cortical representation, M-by-K, M = 128.
    #	sv: scale vector, K-by-1
    #	NORM: (optional) 0=flat, 1=full, .5=partial, (default=.9);
    #	SRF: (optional) sample ripple frequency (ripple resolution)
    #	M0: (optional) original length 
    #	FULLOUT: (optional) overlapped output
    #   BP: bandpass filters indicator, default : 0
    #
    #	COR2AUDS reconstructs auditory spectrum YH from a complex
    #	cortical response Z with respect to scale vector SV. 
    #	CFIL specify the cortical filters. Normaliztion can be selected
    #	out of three style in which FLAT normalization is default. The
    #	default ripple resolution is 24 ch / octave ripple.
    #	See also: AUD2CORS, COR2AUD

    # Author: Powen Ru (powen@isr.umd.edu), NSL, UMD
    # Revision: Taishih Chi (tschi@isr.umd.edu), NSL, UMD
    # Python version:  Nina K. Burns, MITRE

    # target cortical representaion
    [M, K] = np.shape(z)
    if len(sv) != K:
        print('Size not matched')
        sys.exit(0)

    Z_cum = 0
    HH = 0
    dM = np.floor((M-M0)/2)
    M1 = nextpow2(M0)
    M2 = M1 * 2

    if dM:
        z[np.concatenate((np.arange(0,M0+dM),np.arange(M2-dM,M2))),:] \
          = z[np.concatenate((np.arange(dM,M0+dM+dM),np.arange(0,dM))),:]
                                          
    # cumulation
    for k in np.arange(0,K):
        H = gen_corf(sv[k], M1, SRF, [k+BP,K+BP*2])
        R1 = np.fft.fft(z[:, k], M2)
        Z_cum = Z_cum + R1[0:M1] * H
        HH = HH + H**2

    # normalization
    HH = HH * NORM + np.max(HH) * (1 - NORM)
    HH[0] = 2*HH[0]	 # normalization for DC
    Z_cum[0:M0] = Z_cum[0:M0] / HH[0:M0]

    yh = np.fft.ifft(Z_cum, M2)

    if FULLOUT:
        yh = yh[np.concatenate((np.arange(M2-dM,M2),np.arange(0,M0+dM)))]*2
    else:
        yh = yh[0:M0]*2

    return yh


def aud2cor_detect(y, para1, rv, sv, fname, DISP=0):
# AUD2COR (forward) cortical rate-scale representation
#    cr = aud2cor(y, para1, rv, sv, fname, DISP);
#	cr	: cortical representation (4D, scale-rate(up-down)-time-freq.)
#	y	: auditory spectrogram, N-by-M, where
#		N = # of samples, M = # of channels
#	para1 = [paras FULLT FULLX BP]
#	 paras	: see WAV2AUD.
#	 FULLT (FULLX): fullness of temporal (spectral) margin. The value can
#		be any real number within [0, 1]. If only one number was
#		assigned, FULLT = FULLX will be set to the same value.
#	 BP	: pure bandpass indicator
#	 rv	: rate vector in Hz, e.g., 2.^(1:.5:5).
#	 sv	: scale vector in cyc/oct, e.g., 2.^(-2:.5:3).
#	 fname  : cortical representation file, if fname='tmpxxx', no data
#		  will be saved on disk to reduce processing time.
# 	 DISP	: saturation level for color panels. No display if 0.
#		If DISP < 0, every panel will be normalized by its max.
#
#	AUD2COR implements the 2-D wavelet transform
#	possibly executed by the A1 cortex. The auditory
#	spectrogram (Y) is the output generated by the 
#	cochlear model (WAV2AUD) according to the parameter
#	set PARA1. RV (SV) is the characteristic frequencies
#	(ripples) of the temporal (spatial) filters. This
#	function will store the output in a file with a
#	conventional extension .COR. Roughly, one-second
#	signal needs about 22 MB if 8-ms-frame is adopted.
#	Choosing truncated fashion (FULL = 0) will reduce
#	the size to 1/4, which will also reduce runing time
#	to half.
#	See also: WAV2AUD, COR_INFO, CORHEADR, COR_RST

# Author: Powen Ru (powen@isr.umd.edu), NSL, UMD
# Revision: Taishih Chi (tschi@isr.umd.edu), NSL, UMD
# Revision: Elena Grassi (egrassi@umd.edu), NSL, ISR, UMD
# Python version:  Nina K. Burns, MITRE

    if len(para1) < 5:
        FULLT = 0
    else:
        FULLT = para1[4]

    if len(para1) < 6:
        FULLX = FULLT
    else:
        FULLX = para1[5]

    if len(para1) < 7:
        BP = 0
    else:
        BP = para1[6]


    # dimensions
    K1 	= len(rv)	# # of rate channel
    K2	= len(sv)	# # of scale channel
    N, M = np.shape(y)	# dimensions of auditory spectrogram

    # spatial, temporal zeros padding 
    N1 = nextpow2(N)
    N2 = N1*2
    M1 = nextpow2(M)
    M2 = M1*2

    # first fourier transform (w.r.t. frequency axis)
    Y = np.zeros((N2, M1),dtype=np.complex128)
    for n in np.arange(0,N):
        R1 = np.fft.fft(y[n,:], M2)
        Y[n,:] = R1[0:M1]

    # second fourier transform (w.r.t. temporal axis)
    for m in np.arange(0,M1):
        R1 = np.fft.fft(Y[0:N,m],N2)
        Y[:, m] = R1

    paras = para1[0:4]		# parameters for aud. spectrogram
    STF = 1000 / paras[0]	# frame per second
    if (M == 95):
        SRF = 20
    else:
        SRF = 24		# channel per octave (fixed)

    #fout = fopen(fname, 'w')
    #fwrite(fout, [paras(:); K1; K2; rv(:); sv(:); N; M; FULLT; FULLX], ...
    #'float');  

    #TMP = 0        # default: write to HD
    #if len(fname)>=3
    #    TMP = strcmp(fname(1:3), 'tmp');

    # graphics
    #if DISP:
    #    load a1map_a
    #    colormap(a1map)

    #t0 = clock

    # freq. index
    dM   = np.floor(M/2*FULLX)
    mdx1 = np.concatenate((np.arange(0,dM)+M2-dM,np.arange(0,M+dM))).astype(int)
    mdx2 = np.array([0,0,M+1,M+1,0])+dM

    # temp. index
    dN   = np.floor(N/2*FULLT)
    ndx  = np.arange(0,N+2*dN).astype(int)
    ndx1 = ndx
    ndx2 = np.array([0,N+1,N+1,0,0])

    z  = np.zeros((int(N+2*dN), int(M+2*dM)),dtype=np.complex128)
    cr = np.zeros((K2,K1*2,int(N+2*dN),int(M+2*dM)),dtype=np.complex128)
    import matplotlib.pyplot as plt
    
    for rdx in np.arange(1,K1+1):

        # rate filtering
        fc_rt = rv[rdx-1]
        HR = gen_cort_detect(fc_rt, N1, STF, [rdx+BP,K1+BP*2]) # if BP, all BPF 

        for sgn in np.array([1,-1]):

            # rate filtering modification
            if sgn > 0:
                HR = np.concatenate((HR,np.zeros((N1,))))	# SSB -> DSB
            else:
                HR = np.hstack((HR[0],np.conj(np.flipud(HR[1:N2]))))
                HR[N1] = np.abs(HR[N1+1])

            # first inverse fft (w.r.t. time axis)
            z1= np.zeros((N2,M1),dtype=np.complex128)
            for m in np.arange(0,M1):
                z1[:,m]= HR*Y[:,m]

            z1= np.fft.ifft(z1,axis=0)
            z1= z1[ndx1,:]

            for sdx in np.arange(1,K2+1):
                # scale filtering
                fc_sc = sv[sdx-1]
                HS = gen_corf(fc_sc, M1, SRF, [sdx+BP,K2+BP*2]) # if BP, all BPF

                # second inverse fft (w.r.t frequency axis)
                for n in ndx:
                    R1 = np.fft.ifft(z1[n,:]*HS.transpose(),M2)
                    z[n,:] = R1[:,mdx1]	# z: N+2*dN -by- M+2*dM

                # save file
                cr[sdx-1,rdx-1+(sgn==1)*K1,:,:] = z
                #if TMP is False:
                #    nsl.corcplxw(z,fout) # not written yet

                #if DISP:
                #   not written yet
                #   imshow(nsl.cplx_col(z, DISP).transpose())

            if FULLT or FULLX:
                import matplotlib.pyplot as plt
                plt.plot(ndx2, mdx2, 'k--')
        
    return cr

def gen_cort_detect(fc, L, STF, PASS=[1,2]):
# THIS VERSION IS USED FOR IEEE TRANS PAPER ON SPEECH DETECTION
# Python version:  Nina K. Burns, MITRE


    # tonotopic axis
    t = np.arange(0,L).T/STF * fc

    #h = cos(2*pi*t) .* sqrt(t) .* exp(-2*t) * fc
    h = np.sin(2*np.pi*t) * t**2 * np.exp(-3.5*t) * fc

    h = h-np.mean(h)
    H0 = np.fft.fft(h, 2*L)
    A = np.angle(H0[0:L])
    H = np.abs(H0[0:L])
    maxH = np.amax(H)
    maxi = np.argmax(H)
    H = H / np.amax(H)

    # passband
    if PASS[0] == 1:			#lowpass
        sumH = sum(H)
        H[0:maxi] = np.ones(maxi,)
        H = H / np.max(H)		#sum(H) * sumH; # h real-> abs(H) even
        # Make it a min-phase filter (Oppenheim-Shafer 1989 p.784)
        Re_H_hat=np.log(H)
        # real cepstrum. take real to remove numerical error
        c=np.real(np.fft.ifft(np.concatenate((Re_H_hat,Re_H_hat[::-1]))))
        l_min=np.zeros((2*L,))
        l_min[0]= 1
        l_min[L]=1
        l_min[1:L]= 2
        # 2*u[n]-del[n]: see (12.101) p. 794
        # values for n<0 appear in N/2<n<=N-1(or end)
        # 2*u[n]-del[n]= [l[0]=1, 2, ...,l[L-1]=2, 0,...,0]
        h_hat_min=np.multiply(c,l_min) # complex cepstrum
        H_hat_min=np.fft.fft(h_hat_min)
        H= np.multiply(H,np.exp(1j*np.imag(H_hat_min[0:L])))    # H_min phase
    elif PASS[0] == PASS[1]:	# highpass
        sumH = np.sum(H)
        H[maxi:L] = np.ones((L-maxi,))
        H = H / np.max(H)	# sum(H) * sumH
        # Make it a min-phase filter (Oppenheim-Shafer 1989 p.784)
        Re_H_hat= np.log(H) # even
        c=np.real(np.fft.ifft(np.concatenate((Re_H_hat,Re_H_hat[::-1])))) # real cepstrum
        l_min= np.zeros((2*L,))
        l_min[0]=1
        l_min[L]=1
        l_min[1:L]=2
        # 2*u[n]-del[n]: see (12.101) p. 794: values for n<0 appear in N/2<n<=N-1(or end)
        # 2*u[n]-del[n]= [l[0]=1, 2, ...,l[L-1]=2, 0,...,0]
        h_hat_min= np.multiply(c,l_min) # complex cepstrum
        H_hat_min=np.fft.fft(h_hat_min)
        H= np.multiply(H,np.exp(1j*np.imag(H_hat_min[0:L])))  # H_min phase
    else:
        # H = H .* exp(i*A);
        #return a non-causal result!
        H = H+1j*0.00000000001

    return H


