import numpy as np
import math
from scipy import stats
from PyEigerData import GeneralData

# convert the data in self.ProccessedData to pole figure

class GIXS(GeneralData):
    def __init__(self):
        super().__init__()
        self.Description = 'GIXS'
        self.ReflectionCenterX = None
        self.ReflectionCenterY = None
        self.IncidentAngleDeg = None                                                        # in deg
        self.DataReduceRatio = 0.9
        self.Reduction = None
        
    @property                                                                               # properties depend on other properties
    def IncidentAngle(self):                                                                # in rad
        if self.IncidentAngleDeg is None:
            return None
        else:
            return math.radians(self.IncidentAngleDeg)

    def reduction(self):
                                                                                            # alternative input: incident angle or specular center
        if (self.IncidentAngleDeg is None) & (self.ReflectionCenterX is None):
            print('No information of the incident angle or the specular point.')
        elif self.IncidentAngleDeg is None:                                                 # case 1: input specular center
            self.ReflectionCenterX = self.BeamCenterX                                       #  notice: ignore the tilted angle that sample rotate along the sample x axis
            VerticalShift  = - (self.ReflectionCenterY - self.BeamCenterY) *self.YPixelSize # Notice: y axis direction
            self.IncidentAngle = math.atan(VerticalShift/self.DetectorDistance)/2
        elif self.ReflectionCenterX is None:                                                # case 2: input only incident angle
            VerticalShift = self.DetectorDistance * math.atan(2*self.IncidentAngle)
            VerticalShiftPixel = VerticalShift/self.YPixelSize
            self.ReflectionCenterX = self.BeamCenterX
            self.ReflectionCenterY = self.BeamCenterY - VerticalShiftPixel                  # Notice: y axis direction
    
        
        # detemine the O point on the screen
        OX = (self.ReflectionCenterX + self.BeamCenterX)/2
        OY = (self.ReflectionCenterY + self.BeamCenterY)/2

        # create X and Y index matrix
        [YIdxMatrix,XIdxMatrix] = np.where(np.ones([self.YPixelsInDetector,self.XPixelsInDetector],dtype=bool))
        YIdxMatrix = YIdxMatrix.reshape(self.YPixelsInDetector,self.XPixelsInDetector)
        XIdxMatrix = XIdxMatrix.reshape(self.YPixelsInDetector,self.XPixelsInDetector)

        # create the matries record the pixel distance of x and y to O point
        YPixelDistToOMatrix = YIdxMatrix-OY
        XPixelDistToOMatrix = XIdxMatrix-OX
        # create the matries recorded the real distance of x and y to O point
        YDistToOMatrix = YPixelDistToOMatrix * self.YPixelSize
        XDistToOMatrix = XPixelDistToOMatrix * self.XPixelSize

        # create the matrix record the distacne of the scattering point (sample) to each pixel 
        YPixelDistToDBMatrix = YIdxMatrix-self.BeamCenterY                                  # to direct beam DB
        XPixelDistToDBMatrix = XIdxMatrix-self.BeamCenterX                                  # to direct beam DB
        YDistToDBMatrix = YPixelDistToDBMatrix * self.YPixelSize
        XDistToDBMatrix = XPixelDistToDBMatrix * self.XPixelSize
        DistToDBMatrix = np.sqrt(XDistToDBMatrix**2 + YDistToDBMatrix**2)
        DistToSample = np.sqrt(DistToDBMatrix**2 + self.DetectorDistance**2)

        # create 2th matrix defined by Fig.4 Zhang Jiang, J. Appl. Cryst. (2015). 48, 917–926
        twoth = np.arctan(XDistToOMatrix/self.DetectorDistance)        # singed
        twoTh = np.arccos(self.DetectorDistance/DistToSample)          # un signed
        # create alpha_f matrix defined by Fig.4 Zhang Jiang, J. Appl. Cryst. (2015). 48, 917–926
        OToDBDist = np.sqrt( ((OY - self.BeamCenterY)*self.YPixelSize)**2 + ((OX - self.BeamCenterX)*self.XPixelSize)**2 )
        SampleToODist = np.sqrt(OToDBDist**2 + self.DetectorDistance**2)
        Temp = np.sqrt(SampleToODist**2 + XDistToOMatrix**2)
        alphaf = np.arccos ((YDistToOMatrix**2 - DistToSample**2 - Temp**2) / (-2 * DistToSample*Temp)) # un signed
        alphaf[YIdxMatrix>OY] = -1*alphaf[YIdxMatrix>OY] # signed
     
        # create NaN ROI
        if self.ROI is None:
            NaNROI = self.convLogical2NAN(np.ones([self.YPixelsInDetector,self.XPixelsInDetector],dtype=bool))
        else:
            NaNROI = self.convLogical2NAN(self.ROI)

        # create qz, qx, qy matries and using angstrom as length unit
        k = 2*math.pi/(self.Wavelength*1E10)
        qz = k*np.sin(alphaf) + k*np.sin(self.IncidentAngle)
        qx = k*np.cos(alphaf)*np.cos(twoth) - k*np.cos(self.IncidentAngle)
        qy = k*np.cos(alphaf)*np.sin(twoth)
        qr = np.sqrt(qx**2 + qy**2)
        qr[XIdxMatrix<OX] = qr[XIdxMatrix<OX]*-1

        # apply nan mask on qx qy qz qr
        qz = qz * NaNROI
        qx = qx * NaNROI
        qy = qy * NaNROI
        qr = qr * NaNROI

        # re-create qr and qz axis using data reduce ratio
        qzmin = np.nanmin(qz)
        qzmax = np.nanmax(qz)
        qzCen = (qzmax + qzmin)/2
        qzHalfRange = (qzmax - qzmin)/self.DataReduceRatio/2
        qzmin = qzCen - qzHalfRange
        qzmax = qzCen + qzHalfRange
        qrmin = np.nanmin(qr)
        qrmax = np.nanmax(qr)
        qrCen = (qrmax + qrmin)/2
        qrHalfRange = (qrmax - qrmin)/self.DataReduceRatio/2
        qrmin = qrCen - qrHalfRange
        qrmax = qrCen + qrHalfRange

        qzAxis, qzInterval = np.linspace(qzmin,qzmax,self.YPixelsInDetector,retstep=True) # extned range and keep pixel number
        qrAxis, qrInterval = np.linspace(qrmin,qrmax,self.XPixelsInDetector,retstep=True) # extned range and keep pixel number

        qzBoundaryList = np.append(qzAxis, qzAxis[-1] + qzInterval) - qzInterval/2 # make the points on qzAxis at the center of each interval
        qrBoundaryList = np.append(qrAxis, qrAxis[-1] + qrInterval) - qrInterval/2 # make the points on qrAxis at the center of each interval

        # usage: ret = scipy.stats.binned_statistic_2d(x, y, values, statistic='mean', bins=10, range=None, expand_binnumbers=False)
        # reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic_2d.html
        # reference: https://stackoverflow.com/questions/45246811/python-binned-statistic-2d-mean-calculation-ignoring-nans-in-data
        # the statistic part can be defined by numpy.nanmean, it works simply but very slow, the command is listed below:
        # ret = stats.binned_statistic_2d(qr.flatten(), qz.flatten(), self.ProcessedData.flatten(), np.nanmean, bins=[qrBoundaryList, qzBoundaryList], expand_binnumbers=True)
        # when nans present in x or y, the point will be ignore where the bin in result will out of range and the count will not be count in
        qrTemp = qr.flatten()
        qzTemp = qz.flatten()
        DataTemp = self.ProcessedData.flatten()
        SkipIdx = np.isnan(DataTemp)
        
        qrTemp = np.delete(qrTemp,SkipIdx)
        qzTemp = np.delete(qzTemp,SkipIdx)
        DataTemp = np.delete(DataTemp,SkipIdx)
        
        ret = stats.binned_statistic_2d(qzTemp, qrTemp, DataTemp, statistic='mean', bins=[qzBoundaryList, qrBoundaryList], expand_binnumbers=True)
        # note: x using qz and y using qx, no idea what happened

        self.Reduction = {}
        self.Reduction['Data'] = ret.statistic
        self.Reduction['XAxis'] = qrAxis
        self.Reduction['YAxis'] = qzAxis
        
