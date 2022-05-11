## usage:
## from PyEigerData import EigerData
## handle = EigerData() to create toe object handle

import h5py
import hdf5plugin
import os
import numpy as np
import math

class EigerData:
    def __init__(self):
        self.Buffer = None # for test
        self.Description = 'EigerData'
        self.MasterFF = None 
        self.MasterFN = None
        self.MasterFP = None
        self.BitDepthImage = None
        self.XPixelsInDetector = None
        self.YPixelsInDetector = None
        self.CountTime = None
        self.DetectorDistance = None
        self.XPixelSize = None
        self.YPixelSize = None
        self.Wavelength = None
        self.BeamCenterX = None
        self.BeamCenterY = None
        self.PixelMask = None
        self.ContainFrames = None # Contain frame number in this h5 package
        
        ## note that frame SN strart from 1 but frame idx start from 0

    def open(self,MasterFI): # master file information
        ## check file exist or not
        ## if file doesn't exist, return False
        if os.path.exists(MasterFI):
            self.MasterFP = os.path.abspath(MasterFI)
            self.MasterFF, self.MasterFN = os.path.split(self.MasterFP)          
        else:
            print('File doesn'' exist.')
            return False

        ## read header
        self.__readHeader()

    def __readHeader(self):
        FObj = h5py.File(self.MasterFP,'r')
        
        self.BitDepthImage = FObj['/entry/instrument/detector/bit_depth_image'][()]
        self.XPixelsInDetector = FObj['/entry/instrument/detector/detectorSpecific/x_pixels_in_detector'][()]
        self.YPixelsInDetector = FObj['/entry/instrument/detector/detectorSpecific/y_pixels_in_detector'][()]
        self.CountTime = FObj['/entry/instrument/detector/count_time'][()]
        self.DetectorDistance = FObj['/entry/instrument/detector/detector_distance'][()]
        self.XPixelSize = FObj['/entry/instrument/detector/x_pixel_size'][()]
        self.YPixelSize = FObj['/entry/instrument/detector/y_pixel_size'][()]
        self.Wavelength = FObj['/entry/instrument/beam/incident_wavelength'][()]*1E-10 # convert from A to meter
        self.BeamCenterX = FObj['/entry/instrument/detector/beam_center_x'][()]
        self.BeamCenterY = FObj['/entry/instrument/detector/beam_center_y'][()]
        self.PixelMask = FObj['/entry/instrument/detector/detectorSpecific/pixel_mask'][()]
        self.PixelMask = self.PixelMask.astype(bool) # convert the mask to logical array
        
        DataGroupNameList = np.array(FObj['/entry/data'])
        NDataGroup = len(DataGroupNameList)
        ContainFramesInExtLink = []
        for DataGroupIdx in range(0,NDataGroup):
            Temp = FObj['entry/data'][DataGroupNameList[DataGroupIdx]].shape[0] # DataShape: (frame,x pixel, y pixel)
            ContainFramesInExtLink.append(Temp)
            
        self.ContainFramesInExtLink = ContainFramesInExtLink
        self.ContainFrames = sum(ContainFramesInExtLink)
        FObj.close()

    def readFrame(self,ReqSN):
        ## basic function for read single data 
        ## ReqSN: require frame SN
        FObj = h5py.File(self.MasterFP,'r')

        ## find ReqSN in links
        DataGroupNameList = np.array(FObj['/entry/data'])
        NDataGroup = len(DataGroupNameList)
        NextDagaGroupStartSN = 1
        for DataGroupIdx in range(0,NDataGroup):
            ContainFramesInExtLink = FObj['entry/data'][DataGroupNameList[DataGroupIdx]].shape[0] # DataShape: (frame,x pixel, y pixel)
            StartSN = NextDagaGroupStartSN; # Start SN in this datagroup
            EndSN = StartSN + ContainFramesInExtLink - 1 # End SN in this datagroup
            NextDagaGroupStartSN = EndSN + 1 # Start SN in next datagroup
            
            if (ReqSN >= StartSN) & (ReqSN <= EndSN):
                FrameSNInExtLink = ReqSN - StartSN + 1
                FrameIdxInExtLink = FrameSNInExtLink - 1
                break

        SingleFrameData = FObj['entry/data/'+ DataGroupNameList[DataGroupIdx]][FrameIdxInExtLink,:,:]

        FObj.close()
        return SingleFrameData


class GeneralData(EigerData):
    def __init__(self):
        super().__init__()
        self.Description = 'GeneralData'
        self.ROI = None
        self.RawData = None
        self.RawDataFrameSN = None
        self.ProcessedData = None
        self.ProcessedDataFrameNum = None
        self.NormBackground =None
        
    def loadData(self,ReqSNs):
        # load data to self.RawData
        # No return
        if not isinstance(ReqSNs,list):
            ReqSNs = [ReqSNs]

        ReqNum = len(ReqSNs)
        buffer = np.zeros([ReqNum,self.YPixelsInDetector,self.XPixelsInDetector])
        for ReqIdx in range(0,ReqNum):
            buffer[ReqIdx,:,:] = self.readFrame(ReqSNs[ReqIdx])

        self.RawData = buffer
        self.RawDataFrameSN = ReqSNs

    def convLogical2NAN(self,LogicalROIArray):
        ## convert logical ROI to NAN ROI.
        ## In NAN ROI, interesting part are 1 and others are NaN
        NANROI = np.ones(LogicalROIArray.shape)
        NANROI[LogicalROIArray == 0] = np.nan
        return NANROI

    def convMask2ROI(self,LogicalMaskArray):
        ## convert logical mask to ROI
        LogicalROI = ~LogicalMaskArray
        return LogicalROI

    def normalize(self):
        ## normaliz data in self.RawData and store in ProcessedData
        ## apply nan ROI when ROI exist
        self.ProcessedDataFrameNum = len(self.RawDataFrameSN)
        buffer = np.sum(self.RawData,axis = 0)/self.CountTime/self.ProcessedDataFrameNum
        self.ProcessedData = buffer

    def applyROI(self):
        ## check ROI exist or not
        if self.ROI is None:
            print('ROI is empty.')
            return
        else:
            self.ProcessedData = self.ProcessedData*self.convLogical2NAN(self.ROI)

    def setBackground(self,DataPackage):
        ## check existence of ProcessedData in data package of background
        if DataPackage.ProcessedData is None:
            print('Processed data does not exist in input data package.')
            return
        self.NormBackground = DataPackage.ProcessedData

    def suppressBackground(self,DataCompensationFactor,BGCompensationFactor):
        ## check existence of ProcessedData
        if self.NormBackground is None:
            print('Background configuration does not be completed.')
            return
        self.ProcessedDataFrameNum = len(self.RawDataFrameSN)
        buffer = self.RawData/DataCompensationFactor - self.NormBackground/BGCompensationFactor*self.CountTime
        buffer = np.sum(buffer,axis = 0)/self.CountTime/self.ProcessedDataFrameNum
        self.ProcessedData = buffer

    def processData():
        tag_norm = False
        tag_ROI = False
        tag_suppress = False
        DataCompensationFactor = None
        BGCompensationFactor = None
        nargin = len(args)
        if nargin == 0:
            print('Option:')
            print('\t\'norm\'\t\t: normalize loaded data by count time and frames.')
            print('\t\'ROI\'\t\t: apply ROI.')
            print('\t\'suppress\'\t: suppress background using Data improt by _.setBackground')

            print('Example 1:')
            print('\t_.DataProcess(\'norm\')')
            print('\tNormalize the data.')
            print('Example 2:')
            print('\t_.DataProcess(\'norm\',\'ROI\')')
            print('\tNormalize the data and apply ROI.')
            print('Example 3:')
            print('\t_.DataProcess(\'supress\',0.8,0.9)')
            print('\tSuppress background with sample trasmittance 0.8 and buffer transmittance 0.9.')
            print('\t* This function includes normalization process after background supress.')
            print('Example 4:')
            print('\t_.DataProcess(\'supress\',0.8,0.9,\'ROI\')')
            print('\tSuppress background and apply ROI.')
        else:
            if 'norm' in args:
                tag_norm = True
            if 'ROI' in args:
                tag_ROI = True
            if 'suppress' in args:
                tag_suppress = True
                tag_norm = False ## suppress include normlized
                for argIdx in range(0,nargin):
                    if args[argIdx] == 'suppress':
                        DataCompensationFactor = args[argIdx+1]
                        BGCompensationFactor = args[argIdx+2]


        ## process
        if tag_norm:
            self.normalize()
        if tag_suppress:
            self.suppressBackground(DataCompensationFactor,BGCompensationFactor)
        if tag_ROI:
            self.applyROI()
