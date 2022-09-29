% for generating train/test splits
% dataset='...'
nr=100;
load(dataset)
perms=zeros(nr,size(x,1));
for i=1:nr
    perms(i,:)=randperm(size(x,1));
end
save([dataset '_perms'],'perms');