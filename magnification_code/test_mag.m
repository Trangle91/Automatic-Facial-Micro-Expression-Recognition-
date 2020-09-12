clear;
addpath('EVM_Matlab')
addpath('EVM_Matlab//matlabPyrTools')
dataDir = 'test_data';
result_Dir = 'out_dir';
mkdir(result_Dir)
inFile = fullfile(dataDir , 'test.avi')

amplify_spatial_lpyr_temporal_iir(inFile, result_Dir , 10 , 16, 0.4, 0.05, 0.1);