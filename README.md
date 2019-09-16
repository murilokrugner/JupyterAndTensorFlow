#Como Construir uma Rede Neural para Reconhecer Dígitos Manuscritos com o TensorFlow

![Example image](https://assets.digitalocean.com/articles/handwriting_tensorflow_python3/wBCHXId.png)

# Exemplo real:

![Example code](https://github.com/murilokrugner/Reconhecendo-Digitos-Manuscritos-com-TensorFlow/blob/master/img/example-code.png)

# Passo 2 — Importando o Dataset MNIST
O dataset que estaremos utilizando neste tutorial é chamado de dataset MNIST, e ele é um clássico na comunidade de machine learning. Este dataset é composto de imagens de dígitos manuscritos, com 28x28 pixels de tamanho. Aqui estão alguns exemplos dos dígitos incluídos no dataset:
Ao ler os dados, estamos usando one-hot-encoding para representar os rótulos (o dígito real desenhado, por exemplo “3”) das imagens. O one-hot-encoding utiliza um vetor de valores binários para representar valores numéricos ou categóricos. Como nossos rótulos são para os dígitos de 0 a 9, o vetor contém dez valores, um para cada dígito possível. Um desses valores é definido como 1, para representar o dígito nesse índice do vetor, e o restante é difinido como 0. Por exemplo, o dígito 3 é representado usando o vetor [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]. Como o valor no índice 3 está armazenado como 1, o vetor representa o dígito 3.

Para representar as imagens, os 28x28 pixels são achatados em um vetor 1D com 784 pixels de tamanho. Cada um dos 784 pixels que compõem a imagem é armazenado como um valor entre 0 e 255. Isso determina a escala de cinza do pixel, pois nossas imagens são apresentadas apenas em preto e branco. Portanto, um pixel preto é representado por 255 e um pixel branco por 0, com os vários tons de cinza em algum lugar entre eles.

Podemos usar a variável mnist para descobrir o tamanho do dataset que acabamos de importar. Observando os num_examples para cada um dos três subconjuntos, podemos determinar que o dataset foi dividido em 55.000 imagens para treinamento, 5000 para validação e 10.000 para teste. Adicione as seguintes linhas ao seu arquivo:

```
n_train = mnist.train.num_examples # 55,000
n_validation = mnist.validation.num_examples # 5000
n_test = mnist.test.num_examples # 10,000
```
# Passo 3 — Definindo a Arquitetura da Rede Neural
A arquitetura da rede neural refere-se a elementos como o número de camadas na rede, o número de unidades em cada camada e como as unidades são conectadas entre as camadas. Como as redes neurais são vagamente inspiradas no funcionamento do cérebro humano, aqui o termo unidade é usado para representar o que seria biologicamente um neurônio. Assim como os neurônios transmitem sinais pelo cérebro, as unidades tomam alguns valores das unidades anteriores como entrada, realizam uma computação e, em seguida, transmitem o novo valor como saída para outras unidades. Essas unidades são colocadas em camadas para formar a rede, iniciando no mínimo com uma camada para entrada de valores e uma camada para valores de saída. O termo hidden layer ou camada oculta é usado para todas as camadas entre as camadas de entrada e saída, ou seja, aquelas “ocultas” do mundo real.

Arquiteturas diferentes podem produzir resultados drasticamente diferentes, já que o desempenho pode ser pensado como uma função da arquitetura entre outras coisas, como os parâmetros, os dados e a duração do treinamento.

Adicione as seguintes linhas de código ao seu arquivo para armazenar o número de unidades por camada nas variáveis globais. Isso nos permite alterar a arquitetura de rede em um único lugar e, no final do tutorial, você pode testar por si mesmo como diferentes números de camadas e unidades afetarão os resultados de nosso modelo:

```
n_input = 784   # input layer (28x28 pixels)
n_hidden1 = 512 # 1st hidden layer
n_hidden2 = 256 # 2nd hidden layer
n_hidden3 = 128 # 3rd hidden layer
n_output = 10   # output layer (0-9 digits)
```

Outros elementos da rede neural que precisam ser definidos aqui são os hiperparâmetros. Ao contrário dos parâmetros que serão atualizados durante o treinamento, esses valores são definidos inicialmente e permanecem constantes durante todo o processo. No seu arquivo, defina as seguintes variáveis e valores:

```
learning_rate = 1e-4
n_iterations = 1000
batch_size = 128
dropout = 0.5
```

A taxa de aprendizado, learningrate, representa o quanto os parâmetros serão ajustados em cada etapa do processo de aprendizado. Esses ajustes são um componente-chave do treinamento: depois de cada passagem pela rede, ajustamos os pesos ligeiramente para tentar reduzir a perda. Taxas de aprendizado maiores podem convergir mais rapidamente, mas também têm o potencial de ultrapassar os valores ideais à medida que são atualizados. O número de iterações, niterations, refere-se a quantas vezes passamos pela etapa de treinamento e o tamanho do lote ou batch_size se refere a quantos exemplos de treinamento estamos usando em cada etapa. A variável dropout representa um limiar no qual eliminamos algumas unidades aleatoriamente. Estaremos usando dropout em nossa última camada oculta para dar a cada unidade 50% de chance de ser eliminada em cada etapa de treinamento. Isso ajuda a evitar o overfitting.

# Passo 4 — Construindo o Gráfico do TensorFlow

```
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])
keep_prob = tf.placeholder(tf.float32) 
```

O único parâmetro que precisa ser especificado em sua declaração é o tamanho dos dados os quais estaremos alimentando. Para X usamos um formato [None, 784], onde None representa qualquer quantidade, pois estaremos alimentando em um número indefinido de imagens de 784 pixels. O formato de Y é [None, 10] pois iremos usá-lo para um número indefinido de saídas de rótulo, com 10 classes possíveis. O tensor keep_prob é usado para controlar a taxa de dropout, e nós o inicializamos como um placeholder ao invés de uma variável imutável porque queremos usar o mesmo tensor tanto para treinamento (quando dropout é definido para 0.5) quanto para testes (quando dropout é definido como 1.0).

Os parâmetros que a rede atualizará no processo de treinamento são os valores weight e bias, portanto, precisamos definir um valor inicial em vez de um placeholder vazio. Esses valores são essencialmente onde a rede faz seu aprendizado, pois são utilizados nas funções de ativação dos neurônios, representando a força das conexões entre as unidades.

Como os valores são otimizados durante o treinamento, podemos defini-los para zero por enquanto. Mas o valor inicial realmente tem um impacto significativo na precisão final do modelo. Usaremos valores aleatórios de uma distribuição normal truncada para os pesos. Queremos que eles estejam próximos de zero, para que possam se ajustar em uma direção positiva ou negativa, e um pouco diferente, para que gerem erros diferentes. Isso garantirá que o modelo aprenda algo útil. Adicione estas linhas:

```
weights = {
    'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden1], stddev=0.1)),
    'w2': tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2], stddev=0.1)),
    'w3': tf.Variable(tf.truncated_normal([n_hidden2, n_hidden3], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal([n_hidden3, n_output], stddev=0.1)),
}
```

Para o bias ou tendência, usamos um pequeno valor constante para garantir que os tensores se ativem nos estágios iniciais e, portanto, contribuam para a propagação. Os pesos e tensores de bias são armazenados em objetos de dicionário para facilitar o acesso. Adicione este código ao seu arquivo para definir cada bias:

```
biases = {
    'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden1])),
    'b2': tf.Variable(tf.constant(0.1, shape=[n_hidden2])),
    'b3': tf.Variable(tf.constant(0.1, shape=[n_hidden3])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_output]))
}
```

```
layer_1 = tf.add(tf.matmul(X, weights['w1']), biases['b1'])
layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
layer_3 = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
layer_drop = tf.nn.dropout(layer_3, keep_prob)
output_layer = tf.matmul(layer_3, weights['out']) + biases['out']
```

Cada camada oculta executará a multiplicação da matriz nas saídas da camada anterior e os pesos da camada atual e adicionará o bias a esses valores. Na última camada oculta, aplicaremos uma operação de eliminação usando nosso valor keep_prob de 0.5.

Também precisamos escolher o algoritmo de otimização que será usado para minimizar a função de perda. Um processo denominado otimização gradiente descendente é um método comum para encontrar o mínimo (local) de uma função, tomando etapas iterativas ao longo do gradiente em uma direção negativa (descendente). Existem várias opções de algoritmos de otimização de gradiente descendente já implementados no TensorFlow, e neste tutorial vamos usar o otimizador Adam. Isso se estende à otimização de gradiente descendente usando o momento para acelerar o processo através do cálculo de uma média exponencialmente ponderada dos gradientes e usando isso nos ajustes. Adicione o seguinte código ao seu arquivo:

```
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output_layer))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
```

#Passo 5 — Treinando e Testando

O processo de treinamento envolve alimentar o dataset de treinamento através do gráfico e otimizar a função de perda. Toda vez que a rede itera um lote de mais imagens de treinamento, ela atualiza os parâmetros para reduzir a perda, a fim de prever com mais precisão os dígitos exibidos. O processo de teste envolve a execução do nosso dataset de teste através do gráfico treinado e o acompanhamento do número de imagens que são corretamente previstas, para que possamos calcular a precisão.

```
correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
```

Em correct_pred, usamos a função arg_max para comparar quais imagens estão sendo previstas corretamente observando output_layer (predições) e Y (labels), e usamos a função equal para retornar isso como uma lista de Booleanos. Podemos, então, converter essa lista em floats e calcular a média para obter uma pontuação total da precisão.

```
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
```
A essência do processo de treinamento em deep learning é otimizar a função de perda. Aqui, pretendemos minimizar a diferença entre os rótulos previstos das imagens e os rótulos verdadeiros das imagens. O processo envolve quatro etapas que são repetidas para um número definido de iterações:

Propagar valores para frente através da rede
Computar a perda
Propagar valores para trás pela rede
Atualizar parâmetros

```
for i in range(n_iterations):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={X: batch_x, Y: batch_y, keep_prob:dropout})

    # print loss and accuracy (per minibatch)
    if i%100==0:
        minibatch_loss, minibatch_accuracy = sess.run([cross_entropy, accuracy], feed_dict={X: batch_x, Y: batch_y, keep_prob:1.0})
        print("Iteration", str(i), "\t| Loss =", str(minibatch_loss), "\t| Accuracy =", str(minibatch_accuracy))
```

Quando o treinamento estiver concluído, podemos executar a sessão nas imagens de teste. Desta vez estamos usando uma taxa de dropout keep_prob de 1.0 para garantir que todas as unidades estejam ativas no processo de teste.

```
test_accuracy = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob:1.0})
print("\nAccuracy on test set:", test_accuracy)
```

Você verá uma saída semelhante à seguinte, embora os resultados individuais de perda e precisão possam variar um pouco:

```
Iteration 0     | Loss = 3.67079    | Accuracy = 0.140625
Iteration 100   | Loss = 0.492122   | Accuracy = 0.84375
Iteration 200   | Loss = 0.421595   | Accuracy = 0.882812
Iteration 300   | Loss = 0.307726   | Accuracy = 0.921875
Iteration 400   | Loss = 0.392948   | Accuracy = 0.882812
Iteration 500   | Loss = 0.371461   | Accuracy = 0.90625
Iteration 600   | Loss = 0.378425   | Accuracy = 0.882812
Iteration 700   | Loss = 0.338605   | Accuracy = 0.914062
Iteration 800   | Loss = 0.379697   | Accuracy = 0.875
Iteration 900   | Loss = 0.444303   | Accuracy = 0.90625

Accuracy on test set: 0.9206
```

Adicionando a imagem

```
img = np.invert(Image.open("test_img.png").convert('L')).ravel()
```

A função open da bibliotecaImage carrega a imagem de teste como um array 4D contendo os três canais de cores RGB e a transparência Alpha. Esta não é a mesma representação que usamos anteriormente ao ler o dataset com o TensorFlow, portanto, precisamos fazer algum trabalho extra para corresponder ao formato.

Primeiro, usamos a função convert com o parâmetro L para reduzir a representação 4D RGBA para um canal de cor em escala de cinza. Aarmazenamos isso como um array numpy e o invertemos usando np.invert, porque a matriz atual representa o preto como 0 e o branco como 255, porém, precisamos do oposto. Finalmente, chamamos ravel para achatar o array.

Agora que os dados da imagem estão estruturados corretamente, podemos executar uma sessão da mesma forma que anteriormente, mas desta vez apenas alimentando uma imagem única para teste. Adicione o seguinte código ao seu arquivo para testar a imagem e imprimir o rótulo de saída.

```
[labe main.py]
prediction = sess.run(tf.argmax(output_layer,1), feed_dict={X: [img]})
print ("Prediction for test image:", np.squeeze(prediction))
```

A função np.squeeze é chamada na predição para retornar o único inteiro da matriz (ou seja, para ir de [2] para 2). A saída resultante demonstra que a rede reconheceu essa imagem como o dígito 2.

Resultado:

```
Output
Prediction for test image: 2
```


