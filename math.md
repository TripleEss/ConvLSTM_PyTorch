## RNN

```math
\begin{aligned}
\boldsymbol{h}_t =& \tanh(W_x \boldsymbol{x}_{t} + W_h \boldsymbol{h}_{t-1}+\boldsymbol{b})
\end{aligned}
```

## LSTM

```math
\begin{aligned}
\boldsymbol{f} =& \sigma(W_x^{(f)} \boldsymbol{x}_{t} + W_h^{(f)} \boldsymbol{h}_{t-1} + \boldsymbol{b}^{(f)}) \\
\boldsymbol{i} =& \sigma(W_x^{(i)} \boldsymbol{x}_{i} + W_h^{(i)} \boldsymbol{h}_{t-1} + \boldsymbol{b}^{(i)}) \\
\boldsymbol{o} =& \sigma(W_x^{(o)} \boldsymbol{x}_{t} + W_h^{(o)} \boldsymbol{h}_{t-1} + \boldsymbol{b}^{(o)}) \\
\boldsymbol{g} =& \tanh(W_x^{(g)} \boldsymbol{x}_{t} + W_h^{(g)} \boldsymbol{h}_{t-1} + \boldsymbol{b}^{(g)}) \\
\boldsymbol{c}_t =& \boldsymbol{f} \odot \boldsymbol{c}_{t-1} + \boldsymbol{g}\odot \boldsymbol{i}\\
\boldsymbol{h}_t =& \boldsymbol{o} \odot \tanh(\boldsymbol{c}_t)
\end{aligned}
```

## ConvLSTM

```math
\begin{aligned}
\boldsymbol{f} =& \sigma(W_x^{(f)} * \boldsymbol{x}_{t} + W_h^{(f)} * \boldsymbol{h}_{t-1} + \boldsymbol{b}^{(f)}) \\
\boldsymbol{i} =& \sigma(W_x^{(i)} * \boldsymbol{x}_{i} + W_h^{(i)} * \boldsymbol{h}_{t-1} + \boldsymbol{b}^{(i)}) \\
\boldsymbol{o} =& \sigma(W_x^{(o)} * \boldsymbol{x}_{t} + W_h^{(o)} * \boldsymbol{h}_{t-1} + \boldsymbol{b}^{(o)}) \\
\boldsymbol{g} =& \tanh(W_x^{(g)} * \boldsymbol{x}_{t} + W_h^{(g)} * \boldsymbol{h}_{t-1} + \boldsymbol{b}^{(g)}) \\
\boldsymbol{c}_t =& \boldsymbol{f} \odot \boldsymbol{c}_{t-1} + \boldsymbol{g}\odot \boldsymbol{i}\\
\boldsymbol{h}_t =& \boldsymbol{o} \odot \tanh(\boldsymbol{c}_t)
\end{aligned}
```
