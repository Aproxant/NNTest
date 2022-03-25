using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BackpropagationNN
{
    class Layer
    {
        public int nrIpnut;
        public int nrOutput;

        public ActivationFunc func;

        public double[] inputs;
        public double[] outputs; //a

        public double[] biasDerivative;
        public double[,] weightDerivative;

        public double[,] weightOmega;

        public double[] biasOmega;

        public double[,] deltaWeight;

        public double[,] learningRate;

        public double[] learningRateBiases;

        public double[] sumVector; // z
        public double[,] weights; //W
        public double[,] prevWeights;
        public double[] biases;
        public double[] biasesCorrection;

        private double[][,] pastWeights;

        private static Random rnd=new Random();
        
        public Layer(int _in,int _out,ActivationFunc _func)
        {
            nrIpnut = _in;
            nrOutput = _out;

            func = _func;

            inputs = new double[nrIpnut];
            outputs = new double[nrOutput];
            weights = new double[nrOutput, nrIpnut];

            prevWeights = new double[nrOutput, nrIpnut];

            biases = new double[nrOutput];

            biasesCorrection = new double[nrOutput];

            sumVector = new double[nrOutput];

            biasDerivative = new double[nrOutput];
            weightDerivative = new double[nrOutput, nrIpnut];

            weightOmega= new double[nrOutput, nrIpnut];

            biasOmega= new double[nrOutput];

            deltaWeight = new double[nrOutput, nrIpnut];

            learningRate = new double[nrOutput, nrIpnut];

            learningRateBiases = new double[nrOutput];

            pastWeights = new double[5][,];

            InitWeightsAndBiases();
            for (int i = 0; i < 5; i++)
            {
                pastWeights[i] = new double[nrOutput, nrIpnut];
                for (int j = 0; j < nrOutput; j++)
                    for (int k = 0; k < nrIpnut; k++)
                    {
                        pastWeights[i][j, k] = 0;
                    }
            }
        }
        private double GetRandomNumber(double minimum, double maximum)
        {
            return rnd.NextDouble() * (maximum - minimum) + minimum;
        }
        private void InitWeightsAndBiases()
        {
            for(int i=0;i<nrOutput;i++)
            {
                biases[i] = GetRandomNumber(-0.5, 0.5);
                biasOmega[i] = 0;
                for (int j=0;j<nrIpnut;j++)
                {
                    learningRate[i, j] = 0.1;
                    weights[i, j] = GetRandomNumber(-0.5, 0.5);
                    prevWeights[i, j] = weights[i, j];
                    deltaWeight[i, j] = 0;
                    weightOmega[i, j] = 0;
                    
                }
            }
        }


        public void multiWeightInput()
        {
            for(int i=0;i<weights.GetLength(0);i++)
            {
                outputs[i] = 0;
                for (int j = 0; j < weights.GetLength(1);j++)
                {                    
                    outputs[i]+= weights[i, j] * inputs[j];                        
                }
                outputs[i] += biases[i];
                sumVector[i] = outputs[i]; 
            }
            if (ActivationFunc.Sigmoid == func)
                outputs = ActivationFunctions.Sigmoid(outputs);
            else if (ActivationFunc.Softmax == func)
                outputs = ActivationFunctions.SoftMax(outputs);
        }





    }
}
