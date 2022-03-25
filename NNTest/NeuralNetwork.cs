using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BackpropagationNN
{
    public enum ActivationFunc
    {
        Sigmoid,
        Softmax,
        Tanh
    }
    class NeuralNetwork
    {
        private Layer[] layers;

        private int[] nrLayers;


        public NeuralNetwork(int[] networkStruc)
        {
            layers = new Layer[networkStruc.Length-1];
            nrLayers = new int[networkStruc.Length];
            for(int i=0;i< networkStruc.Length-1;i++)
            {
                nrLayers[i] = networkStruc[i];
                if(i==networkStruc.Length-2)
                    layers[i] = new Layer(networkStruc[i], networkStruc[i + 1], ActivationFunc.Softmax);
                else
                    layers[i] = new Layer(networkStruc[i], networkStruc[i + 1],ActivationFunc.Sigmoid);
            }
            nrLayers[networkStruc.Length - 1] = networkStruc[networkStruc.Length - 1];
        }


       
        private double[] ForwardPass(double[] inputs)
        {
            for(int i=0;i<layers[0].inputs.Length;i++)
            {
                layers[0].inputs[i] = inputs[i];
            }
            for(int i=0;i<layers.Length;i++)
            {
                layers[i].multiWeightInput();
                if (i == layers.Length - 1)
                    break;
                layers[i + 1].inputs = layers[i].outputs;
            }
            //layers[layers.Length - 1].outputs = ActivationFunctions.SoftMax(layers[layers.Length - 1].outputs);
            return layers[layers.Length - 1].outputs;
        }


        /*
        private void GradientCalculation(double[] trainOutput)
        {
            double[,] tmp;
            double[] sigDer;
            for(int i=0;i<trainOutput.Length;i++)
            {

                //last layer 
                layers[layers.Length - 1].biasDerivative= layers[layers.Length - 1].outputs.Select((elem, index) => elem - trainOutput[index]).ToArray();
                layers[layers.Length-1].weightDerivative= MultiplicationFunctions.multiplyVectors(layers[layers.Length - 1].biasDerivative, layers[layers.Length-1].inputs);


                //deeper layers
                for(int j=layers.Length-2;j>=0;j--)
                {
                    tmp= MultiplicationFunctions.multiplyMatrix(MultiplicationFunctions.matrixTranspose(layers[j + 1].weights), MultiplicationFunctions.arrayToMatrix(layers[j + 1].biasDerivative));
                    sigDer = ActivationFunctions.SigmoidDerivative(ActivationFunctions.Sigmoid(layers[j].sumVector)); //add tanh
                    layers[j].biasDerivative= MultiplicationFunctions.matrixToArray(tmp).Select((elem, index) => elem * sigDer[index]).ToArray();
                    layers[j].weightDerivative = MultiplicationFunctions.multiplyVectors(layers[j].biasDerivative, layers[j].inputs);  
                }
                
            }
        }*/
        //changed
        private void GradientCalculation(double[] trainOutput)
        {
            double[,] tmp;
            double[] sigDer;
            for (int i = 0; i < trainOutput.Length; i++)
            {

                //last layer 
                var k=layers[layers.Length - 1].outputs.Select(elem => elem-1).ToArray();
                layers[layers.Length - 1].biasDerivative = trainOutput.Select((elem, index) => elem * k[index]).ToArray();
                layers[layers.Length - 1].weightDerivative = MultiplicationFunctions.multiplyVectors(layers[layers.Length - 1].biasDerivative, layers[layers.Length - 1].inputs);


                //deeper layers
                for (int j = layers.Length - 2; j >= 0; j--)
                {
                    tmp = MultiplicationFunctions.multiplyMatrix(MultiplicationFunctions.matrixTranspose(layers[j + 1].weights), MultiplicationFunctions.arrayToMatrix(layers[j + 1].biasDerivative));
                    sigDer = ActivationFunctions.SigmoidDerivative(ActivationFunctions.Sigmoid(layers[j].sumVector)); //add tanh
                    layers[j].biasDerivative = MultiplicationFunctions.matrixToArray(tmp).Select((elem, index) => elem * sigDer[index]).ToArray();
                    layers[j].weightDerivative = MultiplicationFunctions.multiplyVectors(layers[j].biasDerivative, layers[j].inputs);
                }

            }
        }
        private void weightAdjustment(double momentum) //sprawdzic czy dziala
        {
            for (int i = layers.Length - 1; i >= 0; i--)
            {
                for (int j = 0; j < layers[i].weights.GetLength(0); j++)
                {
                    for (int k = 0; k < layers[i].weights.GetLength(1); k++)
                    {
                        layers[i].prevWeights[j, k] = layers[i].weights[j, k];                       

                        layers[i].deltaWeight[j, k] = (-1) * (layers[i].weightDerivative[j, k] * layers[i].learningRate[j, k]) + (momentum * layers[i].deltaWeight[j, k]);

                        layers[i].weights[j, k] += layers[i].deltaWeight[j, k];
                    }
                }

            }
            
        }
        private void RestoreAndAdjustment(double momentum) //sprawdzic czy dziala
        {
            for (int i = layers.Length - 1; i >= 0; i--)
            {
                for (int j = 0; j < layers[i].weights.GetLength(0); j++)
                {
                    for (int k = 0; k < layers[i].weights.GetLength(1); k++)
                    {
                        layers[i].weights[j, k] = layers[i].prevWeights[j, k];

                        
                        layers[i].deltaWeight[j, k] = (-1) * (layers[i].weightDerivative[j, k] * layers[i].learningRate[j, k]) + (momentum * layers[i].deltaWeight[j, k]);

                        layers[i].weights[j, k] += layers[i].deltaWeight[j, k];

                        layers[i].deltaWeight[j, k] = 0;
                    }
                }

            }
        }
        private void AdjustLearningRate(double phi,double alfaUp,double alfaDown) //fix that
        {
            
            for (int i = layers.Length - 1; i >= 0; i--)
            {
                for (int j = 0; j < layers[i].weights.GetLength(0); j++)
                {
                    layers[i].learningRateBiases[j]= (1 - phi) * layers[i].learningRateBiases[j] + phi * layers[i].biasOmega[j];
                    layers[i].biasOmega[j] = layers[i].learningRateBiases[j];
                    for (int k = 0; k < layers[i].weights.GetLength(1); k++)
                    {
                        if((1 - phi) * layers[i].weightDerivative[j, k] + phi * layers[i].weightOmega[i, j]>0)
                        layers[i].learningRate[j, k] = (1 - phi) * layers[i].weightDerivative[j, k] + phi * layers[i].weightOmega[i, j];
                        layers[i].weightOmega[i, j] = layers[i].learningRate[j, k];
                    }
                }
            }
        }
        public double ErrorOnTest(double[][] input, double[][] output)
        {
            double[] result;
            double error = 0;
            for(int i=0;i<input.Length;i++)
            {
                result=ForwardPass(input[i]);
                error+=ErrorFunctions.Cross_Entropy(output[i], result);
            }
            return error;
        }


        public void Train(Tuple<double[][], double[][], double[][], double[][]> data,int epoch,double learning_rate,double momentum)
        {
            double Error=double.MaxValue,newError;
            double[][] trainInput = data.Item1;
            double[][] trainOutput = data.Item2;
            double[][] testInput = data.Item3;
            double[][] testOutput = data.Item4;
            for(int i=0;i<epoch;i++)
            {
                for(int j=0;j<trainInput.GetLength(0);j++)
                {
                    ForwardPass(trainInput[i]);

                    //BackPropagation
                    GradientCalculation(trainOutput[j]);

                    newError = ErrorFunctions.Cross_Entropy(trainOutput[j], layers[layers.Length - 1].outputs);
                    weightAdjustment(momentum);
                    //WeightAdjustment
                    /*
                    if (newError<1.05*Error)
                    {
                        
                        Error = newError;
                    }
                    else
                    {
                        //restore weights
                        RestoreAndAdjustment(momentum);

                    }
                    */
                    //Console.WriteLine(ErrorFunctions.Cross_Entropy(trainOutput[j], layers[layers.Length - 1].outputs));
                    //Console.WriteLine($"Error is {Error}");
                }
                
                //Console.WriteLine($"Error on train is {ErrorOnTest(testInput, testOutput)}");
            }
        }

    }
}
