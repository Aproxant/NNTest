using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BackpropagationNN
{
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
                layers[i] = new Layer(networkStruc[i], networkStruc[i + 1]);
            }
            nrLayers[networkStruc.Length - 1] = networkStruc[networkStruc.Length - 1];
        }


       
        private void ForwardPass(double[] inputs)
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
        }


        
        private void GradientCalculation(double[] trainOutput)
        {
            // now only for Cross entropy. To do for L2 norm error same procedure.
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
                    sigDer = ActivationFunctions.SigmoidDerivative(ActivationFunctions.Sigmoid(layers[j].sumVector));
                    layers[j].biasDerivative= MultiplicationFunctions.matrixToArray(tmp).Select((elem, index) => elem * sigDer[index]).ToArray();
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


        public void Train(double[][] trainInput, double[][] trainOutput,int epoch,double learning_rate,double momentum)
        {
            double Error=1000,newError;
            bool useMomentum=true;
            for(int i=0;i<epoch;i++)
            {
                for(int j=0;j<trainInput.GetLength(0);j++)
                {
                    ForwardPass(trainInput[i]);

                    //BackPropagation
                    GradientCalculation(trainOutput[j]);

                    newError = ErrorFunctions.Cross_Entropy(trainOutput[j], layers[layers.Length - 1].outputs);

                    //WeightAdjustment
                    if (newError<1.05*Error)
                    {
                        weightAdjustment(momentum);
                    }
                    else
                    {
                        //restore weights
                        RestoreAndAdjustment(momentum);

                    }
                    

                    Error = newError;
                    //ErrorFunctions.Cross_Entropy(trainOutput[j], layers[layers.Length - 1].outputs);
                    Console.WriteLine($"Error is {ErrorFunctions.Cross_Entropy(trainOutput[j], layers[layers.Length - 1].outputs)}");
                }
            }
        }

    }
}
