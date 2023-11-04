clc;
inputs = [0,0,1,50,1,1;
1,1,2,60,1,1;
2,2,0,40,0,0;
0,3,2,70,1,1;
3,4,1,60,1,1;
1,0,1,50,1,1;
0,2,0,40,0,0;
2,1,2,60,1,1;
3,3,1,70,1,1;
1,4,1,60,1,1;
0,0,1,50,1,1;
1,1,2,60,1,1;
2,2,0,40,0,0;
0,3,2,70,1,1;
3,4,1,60,1,1;
1,0,1,50,1,1;
0,2,0,40,0,0;
2,1,2,60,1,1;
3,3,1,70,1,1;
1,4,1,60,1,1];
outputs = [0;
1;
2;
1;
1;
0;
2;
1;
1;
1;
0;
1;
2;
1;
1;
0;
2;
1;
1;
1];
tinput=normalize(inputs,'range')
toutput=normalize(outputs,'range')





w1 = unifrnd(-1,1,6,20);
b1 = unifrnd(-1,1,1,20);
w2 = unifrnd(-1,1,20,1);
rate = 0.2;
for epoch = 1:5000
    for trainIndex = 1:length(tinput)
        tinput1=tinput(trainIndex,:);
        toutput1=toutput(trainIndex,:);
        layer1 = tinput1*w1;
        layer1sigmoid=sigmoid(layer1);
        layer2=layer1sigmoid*w2;
        layer2sigmoid=sigmoid(layer2);
        hata=(toutput1-layer2sigmoid).^2;
        hatam=mean(hata);
        w2d=(2*(layer2sigmoid-toutput1)).*diag(sigmoidd(layer2)).*w2;
        w1d = (w2d.'.*diag(sigmoidd(layer1))*w1.');
        b1d = (w2d.'.*sigmoidd(layer1));
        w2 = w2 - w2d.*rate;
        w1 = w1 - w1d.'.*rate;
        b1 = b1 - b1d.*rate;
    end
end


fprintf('1')
ip = tinput(1,:)
op = toutput(1,:)
feedforward(w1,w2,b1,tinput(1,:),toutput(1,:))

fprintf('112312')
ip = [2 1 2 20 1 11]
op = [2]
feedforward(w1,w2,b1,normalize(ip,'range'),normalize(op,'range'))



function y=sigmoidd(x)
y=sigmoid(x).*(1-sigmoid(x));
end
function y=sigmoid(x)
y=1./(1+exp(-x));
end


function out = feedforward(w1,w2,b1,input,toutput1)
        tinput1=input;
        layer1 = tinput1*w1;
        layer1sigmoid=sigmoid(layer1);
        layer2=layer1sigmoid*w2;
        layer2sigmoid=sigmoid(layer2);
        hata=(toutput1-layer2sigmoid).^2;
        hatam=mean(hata)
        out = layer2sigmoid
end