function C = Crossentropy(y,yhat)
C = sum(-y .* log(yhat) - (1-y) .* log(1-yhat))/length(y);
end

