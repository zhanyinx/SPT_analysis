import matplotlib.pyplot as plt
import numpy as np

def fill_z(a):
    '''Find longest subarray and adjust the rest to its length by filling with zeros'''
    lens = [len(i) for i in a]
    ml = max(lens)
    b = np.zeros((len(lens), ml))
    for i in range(len(lens)):
        for j in range(len(a[i])):
            b[i, j] = a[i][j]
    return b

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, np.sqrt(variance))

clones = ['2B10', '3G4']
noaux = []
wnoaux = []
enoaux = []

aux = []
waux = []
eaux = []
# Load -auxin data, WT
for rep in [0,2]:
    noaux.append(np.loadtxt('//tungsten-nas.fmi.ch/tungsten/scratch/ggiorget/pavel/_analysis_chromatin_dynamics/deepblink/3xCTCF/'+'_1_'+str(rep)+'_dt1_msd_corrected.dat')[:,0])
    wnoaux.append(np.loadtxt('//tungsten-nas.fmi.ch/tungsten/scratch/ggiorget/pavel/_analysis_chromatin_dynamics/deepblink/3xCTCF/'+'_1_'+str(rep)+'_dt1_msd_corrected.dat')[:,1])
    enoaux.append(np.loadtxt('//tungsten-nas.fmi.ch/tungsten/scratch/ggiorget/pavel/_analysis_chromatin_dynamics/deepblink/3xCTCF/'+'_1_'+str(rep)+'_dt1_msd_corrected.dat')[:,2])
# Load +auxin data, mutant
for rep in [0,2]:
    aux.append(np.loadtxt('//tungsten-nas.fmi.ch/tungsten/scratch/ggiorget/pavel/_analysis_chromatin_dynamics/deepblink/3xCTCF/'+'_0_'+str(rep)+'_dt1_msd_corrected.dat')[:,0])
    waux.append(np.loadtxt('//tungsten-nas.fmi.ch/tungsten/scratch/ggiorget/pavel/_analysis_chromatin_dynamics/deepblink/3xCTCF/'+'_0_'+str(rep)+'_dt1_msd_corrected.dat')[:,1])
    eaux.append(np.loadtxt('//tungsten-nas.fmi.ch/tungsten/scratch/ggiorget/pavel/_analysis_chromatin_dynamics/deepblink/3xCTCF/'+'_0_'+str(rep)+'_dt1_msd_corrected.dat')[:,2])
    
noaux2 = fill_z(noaux)
wnoaux2 = fill_z(wnoaux)
enoaux2 = fill_z(enoaux)
aux2 = fill_z(aux)
waux2 = fill_z(waux)
eaux2 = fill_z(eaux)
total_noaux = np.zeros(len(noaux2[0]))
err_noaux = np.zeros(len(noaux2[0]))
for i in range(len(noaux2[0])):
    total_noaux[i], err_noaux[i] = weighted_avg_and_std(noaux2[:,i], wnoaux2[:,i])
    err_noaux[i], grab = weighted_avg_and_std(enoaux2[:,i], wnoaux2[:,i])
total_aux = np.zeros(len(aux2[0]))
err_aux = np.zeros(len(aux2[0]))
for i in range(len(aux2[0])):
    total_aux[i], err_aux[i] = weighted_avg_and_std(aux2[:,i], waux2[:,i])
    err_aux[i], grab = weighted_avg_and_std(eaux2[:,i], waux2[:,i])

plt.figure(figsize=(12,10))
plt.title('Rad21-AID, 3xCTCF,  +/- auxin')
plt.errorbar(10*np.arange(1,len(total_noaux)+1), total_noaux, yerr=err_noaux, label='- aux',elinewidth=2,capsize=4, lw=3)
plt.errorbar(10*np.arange(1,len(total_aux)+1), total_aux, yerr=err_aux, label='+aux ',elinewidth=2,capsize=4, lw=3)

x = np.log10(np.arange(10,60,10))
y = np.log10(total_noaux[0:5])
slope, lgd = np.polyfit(x, y, 1)
plt.gcf().text(x= 0.15, y= 0.8, s = r'upd $\alpha_{-} = $' + '{:.3f}'.format(slope)+r' D$_{-}$ = ' + '{:.4f}'.format(10**lgd) + ' - 10-50 sec')

x = np.log10(np.arange(10,60,10))
y = np.log10(total_aux[0:5])
slope, lgd = np.polyfit(x, y, 1)
plt.gcf().text(x= 0.15, y= 0.775, s = r'upd $\alpha_{+} = $' + '{:.3f}'.format(slope)+r' D$_{+}$ = ' + '{:.4f}'.format(10**lgd) + ' + 10-50 sec')

x = np.log10(np.arange(100,210,10))
y = np.log10(total_noaux[9:20])
slope, lgd = np.polyfit(x, y, 1)
plt.gcf().text(x= 0.15, y= 0.7, s = r'upd $\alpha_{-} = $' + '{:.3f}'.format(slope)+r' D$_{-}$ = ' + '{:.4f}'.format(10**lgd) + ' - 100-200 sec')

x = np.log10(np.arange(100,210,10))
y = np.log10(total_aux[9:20])
slope, lgd = np.polyfit(x, y, 1)
plt.gcf().text(x= 0.15, y= 0.675, s = r'upd $\alpha_{+} = $' + '{:.3f}'.format(slope)+r' D$_{+}$ = ' + '{:.4f}'.format(10**lgd) + ' + 100-200 sec')

x = np.log10(np.arange(300,510,10))
y = np.log10(total_noaux[29:50])
slope, lgd = np.polyfit(x, y, 1)
plt.gcf().text(x= 0.15, y= 0.6, s = r'upd $\alpha_{-} = $' + '{:.3f}'.format(slope)+r' D$_{-}$ = ' + '{:.4f}'.format(10**lgd) + ' - 300-500 sec')

x = np.log10(np.arange(300,510,10))
y = np.log10(total_aux[29:50])
slope, lgd = np.polyfit(x, y, 1)
plt.gcf().text(x= 0.15, y= 0.575, s = r'upd $\alpha_{+} = $' + '{:.3f}'.format(slope)+r' D$_{+}$ = ' + '{:.4f}'.format(10**lgd) + ' + 300-500 sec')
'''
x = np.log10(np.arange(10,60,10))
y = total_noaux[1:6]
slope, lgd = np.polyfit(x, y, 1)
plt.gcf().text(x= 0.15, y= 0.8, s = r'upd $\alpha_{-} = $' + '{:.3f}'.format(slope)+r' D$_{-}$ = ' + '{:.4f}'.format(10**lgd) + ' - 10-50 sec')

x = np.log10(np.arange(10,60,10))
y = total_aux[1:6]
slope, lgd = np.polyfit(x, y, 1)
plt.gcf().text(x= 0.15, y= 0.775, s = r'upd $\alpha_{+} = $' + '{:.3f}'.format(slope)+r' D$_{+}$ = ' + '{:.4f}'.format(10**lgd) + ' + 10-50 sec')

x = np.log10(np.arange(100,210,10))
y = total_noaux[10:21]
slope, lgd = np.polyfit(x, y, 1)
plt.gcf().text(x= 0.15, y= 0.7, s = r'upd $\alpha_{-} = $' + '{:.3f}'.format(slope)+r' D$_{-}$ = ' + '{:.4f}'.format(10**lgd) + ' - 100-200 sec')

x = np.log10(np.arange(100,210,10))
y = total_aux[10:21]
slope, lgd = np.polyfit(x, y, 1)
plt.gcf().text(x= 0.15, y= 0.675, s = r'upd $\alpha_{+} = $' + '{:.3f}'.format(slope)+r' D$_{+}$ = ' + '{:.4f}'.format(10**lgd) + ' + 100-200 sec')

x = np.log10(np.arange(300,510,10))
y = total_noaux[30:51]
slope, lgd = np.polyfit(x, y, 1)
plt.gcf().text(x= 0.15, y= 0.6, s = r'upd $\alpha_{-} = $' + '{:.3f}'.format(slope)+r' D$_{-}$ = ' + '{:.4f}'.format(10**lgd) + ' - 300-500 sec')

x = np.log10(np.arange(300,510,10))
y = total_aux[30:51]
slope, lgd = np.polyfit(x, y, 1)
plt.gcf().text(x= 0.15, y= 0.575, s = r'upd $\alpha_{+} = $' + '{:.3f}'.format(slope)+r' D$_{+}$ = ' + '{:.4f}'.format(10**lgd) + ' + 300-500 sec')
'''

plt.xlim(8,500)
plt.ylim(0.005, 2)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\Delta$t, sec')
plt.ylabel(r'MSD, $\mu m^2$')
plt.legend(loc='lower right')
plt.savefig('//tungsten-nas.fmi.ch/tungsten/scratch/ggiorget/pavel/_analysis_chromatin_dynamics/deepblink/3xCTCF/msd4_3xctcf_sd3.pdf')
plt.show()
plt.clf()
'''
plt.figure(figsize=(12,10))
plt.plot(10*np.arange(1,51), enoaux[0][:50], label='SD 1 clone')
plt.plot(10*np.arange(1,51), enoaux[1][:50], label='SD 2 clone')
plt.plot(10*np.arange(1,51), err_noaux[:50], label='SD average')
plt.plot(10*np.arange(1,51), (err_noaux/np.sqrt(np.sum(wnoaux2, axis=0)))[:50], label='SE average')
plt.legend()
plt.show()
'''