## usage:
## from PyEigerData import EigerData
## handle = EigerData() to create toe object handle

import h5py
import hdf5plugin
import os
import numpy as np
import math
import csv

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
        self.ContainFramesInExtLink = None
        self.LinkData = None
        
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
        ## check file exist or not
        for DataGroupIdx in range(0,len(DataGroupNameList)):
            FF = self.MasterFF
            FN = self.MasterFN.replace('_master','_'+DataGroupNameList[DataGroupIdx])
            FP = os.path.join(FF,FN)
            if not os.path.exists(FP):
                DataGroupNameList = np.delete(DataGroupNameList,range(DataGroupIdx,len(DataGroupNameList)))
                break
            
        self.LinkData = DataGroupNameList
        
        NDataGroup = len(self.LinkData)
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
        DataGroupNameList = self.LinkData
        NDataGroup = len(DataGroupNameList)
        NextDataGroupStartSN = 1
        for DataGroupIdx in range(0,NDataGroup):
            ContainFramesInExtLink = self.ContainFramesInExtLink[DataGroupIdx]
            StartSN = NextDataGroupStartSN; # Start SN in this datagroup
            EndSN = StartSN + ContainFramesInExtLink - 1 # End SN in this datagroup
            NextDataGroupStartSN = EndSN + 1 # Start SN in next datagroup
            
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
        self.DataProcessOption = {'norm':True,'ROI':True,'suppress':False,'DataCompensationFactor':1,'BGCompensationFactor':1}
        
    def loadData(self,ReqSNs):
        # load data to self.RawData
        # No return
        if not isinstance(ReqSNs,list):
            ReqSNs = [ReqSNs]
        ReqSNs = np.array(ReqSNs)

        # check request request SN vialable or not
        if any(ReqSNs > self.ContainFrames):
            print('Request frame SN out of range %d' %(self.ContainFrames))
            return
        

        ReqNum = len(ReqSNs)
        buffer = np.zeros([ReqNum,self.YPixelsInDetector,self.XPixelsInDetector])
        for ReqIdx in range(0,ReqNum):
            buffer[ReqIdx,:,:] = self.readFrame(ReqSNs[ReqIdx])

        self.RawData = buffer
        self.RawDataFrameSN = ReqSNs

                
    def convCSV2Logical(self,CSVFP):
        # load csv data and convert it to a logical matrix
        fid = open(CSVFP)
        csvreader = csv.reader(fid)
        header = next(fid)
        rows = []
        for row in csvreader:
            # row[X,Y,Value]
            rows.append(row)
        fid.close()

        rows = np.array(rows,dtype=int)
        Xidx = rows[:,0].flatten()
        Yidx = rows[:,1].flatten()
        NumTarget = len(Xidx)

        Buffer = np.zeros([self.YPixelsInDetector,self.XPixelsInDetector],dtype = bool)
        for SN in range(0,NumTarget):
            Buffer[Yidx[SN],Xidx[SN]] = True

        return Buffer

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

    def sum(self):
        self.ProcessedDataFrameNum = len(self.RawDataFrameSN)
        buffer = np.sum(self.RawData,axis = 0)
        self.ProcessedData = buffer

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

    def processData(self):
        # process data according options in self.DataProcessOption
        # self.DataProcessOption = {'norm':True,'ROI':False,'suppress':False,'DataCompensationFactor':1,'BGCompensationFactor':1}
        tag_norm = self.DataProcessOption['norm']
        tag_ROI = self.DataProcessOption['ROI']
        tag_suppress = self.DataProcessOption['suppress']
        DataCompensationFactor = self.DataProcessOption['DataCompensationFactor']
        BGCompensationFactor = self.DataProcessOption['BGCompensationFactor']


        if tag_suppress:
            tag_norm = False ## suppress included in normlize


        ## process
        if tag_norm:
            self.normalize()
        if tag_suppress:
            self.suppressBackground(DataCompensationFactor,BGCompensationFactor)
        if tag_ROI:
            self.applyROI()
