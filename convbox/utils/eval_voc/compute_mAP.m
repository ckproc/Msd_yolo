
addpath('/home/ckp/data/PASCAL/VOCdevkit/VOCcode/')

clear VOCopts

VOCopts.classes = {'aeroplane' 'bicycle' 'bird' 'boat' 'bottle' 'bus' 'car' 'cat' 'chair' 'cow' ...
'diningtable' 'dog' 'horse' 'motorbike' 'person' 'pottedplant' 'sheep' 'sofa' 'train' 'tvmonitor'};

VOCopts.nclasses=length(VOCopts.classes);

VOCopts.minoverlap = 0.5;
VOCopts.datadir = '/home/ckp/data/PASCAL/VOCdevkit';
% evalution VOC 2007 test
VOCopts.dataset = 'VOC2007';
VOCopts.testset = 'test';
% evalution VOC 2007 train
% VOCopts.dataset = 'VOC2007';
% VOCopts.testset = 'train';
%VOCopts.resdir = '../../../result/data/';
VOCopts.resdir = '/home/ckp/convbox2/result/data/';

VOCopts.imgsetpath = [VOCopts.datadir '/' VOCopts.dataset '/ImageSets/Main/%s.txt'];
VOCopts.annopath = [VOCopts.datadir '/' VOCopts.dataset '/Annotations/%s.xml'];
VOCopts.detrespath = [VOCopts.resdir '%s_det_' VOCopts.testset '_%s.txt'];

mAP = 0;
for i = 1:VOCopts.nclasses

    cls = VOCopts.classes{i};
    [recall, prec, ap] = VOCevaldet(VOCopts, 'comp4', cls, true);  % compute and display PR
    
    ap, max(prec), max(recall)
    mAP = mAP + ap / 20.0;
    
end

mAP
