function stratBatchIter(targets, approxBatchsize)
	local T = targets:clone()+torch.randn(target:size())*0.001 -- add some jitter for randomness
	local nbatch = torch.floor(T:size(1) / approxBatchsize)
	
	-- these will be 1,2,3,4,..,nbatch,1,2,3,4,...,nbatch,...
	local batchIds = torch.range(1,T:size(1)):viewAs(T)
	for i=1,batchIds:size(1),
		batchIds[i] = 1 + batchIds[i] % nbatch 
	end
	
	-- index of sorted elements 
	local _, ind = torch.sort(T)
	
	-- re-assign to batch
	batchIds:index(1,ind:long()) = batchIds
	
	-- return iterator, which yields a bytetensor for addressing data
	local batchIndex = 0
	return function()
		batchIndex = batchIndex + 1
		if batchIndex == nbatch then
			batchIndex = 1
		end
		return torch.eq(batchIds, batchIndex)
	end
end
