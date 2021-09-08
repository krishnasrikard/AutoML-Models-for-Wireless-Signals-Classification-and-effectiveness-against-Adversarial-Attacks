clc
clear all

printf("Running...\n");
SNRdB=50; % Noise power related
SNR=10^(SNRdB/10);
NSymClass = 500*3;  %500;

% 3 modulation schemes input data
QPSK  = [1+1i 1-1i -1+1i -1-1i]/sqrt(2);
QAM16 = [1+1i 1-1i -1+1i -1-1i 1+3i 1-3i -1+3i -1-3i 3+1i 3-1i -3+1i -3-1i 3+3i 3-3i -3+3i -3-3i]/sqrt(10);
QAM64 = [
    -7-7i -7-5i -7-3i -7-1i -7+1i -7+3i -7+5i -7+7i -5-7i -5-5i -5-3i -5-1i -5+1i -5+3i -5+5i -5+7i ...
    -3-7i -3-5i -3-3i -3-1i -3+1i -3+3i -3+5i -3+7i -1-7i -1-5i -1-3i -1-1i -1+1i -1+3i -1+5i -1+7i ...
    1-7i  1-5i  1-3i  1-1i  1+1i  1+3i  1+5i  1+7i  3-7i  3-5i  3-3i  3-1i  3+1i  3+3i  3+5i  3+7i ...
    5-7i  5-5i  5-3i  5-1i  5+1i  5+3i  5+5i  5+7i  7-7i  7-5i  7-3i  7-1i  7+1i  7+3i  7+5i  7+7i]/sqrt(42);
InpSym = [];

% Length of the channel
ChannelLength = 2;
% The two taps are given powers 0.8 and 0.2; the sum of powers of channel =
% 1, 3 taps powers we can take as 0.75, 0.2, 0.05, basically ensure sum of tap powers = 1
Channel = [sqrt(0.8) sqrt(0.2)].*(randn(1,ChannelLength)+1i*randn(1,ChannelLength)/sqrt(2)); % complex channel generation (each tap is complex normal random variable)
printf("Channel: (%f)+i(%f)\n",Channel(1),Channel(2));
Symlength = 100;
for n=0:NSymClass-1
    if(mod(n+1,500)==0)
      printf("iteration: %d\n",n+1);
    end
    % taking X as data, by randomly taking from vectors QPSK, 16QAM and
    % 64-QAM mentioned above
    InpSym(3*n+1,:) = QPSK(randi([1,length(QPSK)],Symlength,1));
    InpSym(3*n+2,:) = QAM16(randi([1,length(QAM16)],Symlength,1));
    InpSym(3*n+3,:) = QAM64(randi([1,length(QAM64)],Symlength,1));       

    % output signal generation by a) convolving with channel and 
    % b) then adding noise
    OutSym(3*n+1,:)=conv(InpSym(3*n+1,:),Channel)+sqrt(1/(2*SNR))*(randn(1,Symlength+ChannelLength-1)+1i*randn(1,Symlength+ChannelLength-1));
    OutSym(3*n+2,:)=conv(InpSym(3*n+2,:),Channel)+sqrt(1/(2*SNR))*(randn(1,Symlength+ChannelLength-1)+1i*randn(1,Symlength+ChannelLength-1));
    OutSym(3*n+3,:)=conv(InpSym(3*n+3,:),Channel)+sqrt(1/(2*SNR))*(randn(1,Symlength+ChannelLength-1)+1i*randn(1,Symlength+ChannelLength-1));
    
    % Assigning labels to generated data
    Labels(3*n+1)=1;
    Labels(3*n+2)=2;
    Labels(3*n+3)=3;

end
% SaVING dATA
%[indexes] = randperm(3*NSymClass);
[indexes] = [1:3*NSymClass];%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

TrainingDataSym = OutSym(indexes(1:2*NSymClass),:);
TrainingLabels = Labels(indexes(1:2*NSymClass));

Id = NSymClass+1:3*NSymClass; %Just some identity
TrainingData=[Id.' TrainingDataSym TrainingLabels.'];
csvwrite('trainDataLabels.csv',TrainingData);

TestingData = OutSym(indexes(2*NSymClass+1:end),:);
Id_Test=1:NSymClass; % Just some index
csvwrite('testData.csv',[Id_Test.' TestingData]);

TestingLabels = Labels(indexes(2*NSymClass+1:end));
csvwrite('testLabels.csv',[Id_Test.' TestingLabels.']);

DummySubmission = 2*ones(NSymClass,1); %randi([1,3],NSymClass,1);
csvwrite('sampleSubmission.csv',[Id_Test.' DummySubmission]);

printf("Over");