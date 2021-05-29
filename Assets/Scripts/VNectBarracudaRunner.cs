using UnityEngine;
using UnityEngine.UI;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using Unity.Barracuda;

/// <summary>
/// Define Joint points
/// </summary>
public class VNectBarracudaRunner : MonoBehaviour
{
    /// <summary>
    /// Neural network model
    /// </summary>
    public NNModel NNModel;

    public WorkerFactory.Type WorkerType = WorkerFactory.Type.Auto;
    public bool Verbose = true;

    public VNectModel VNectModel;

    public VideoCapture videoCapture;

    private Model _model;
    private IWorker _worker;

    /// <summary>
    /// Coordinates of joint points
    /// </summary>
    private VNectModel.JointPoint[] jointPoints;

    private List<VNectModel.JointPointCache[]> cachedJointPoints = new List<VNectModel.JointPointCache[]>();
    private bool useCache;
    private int cacheCursor = 0;
    /// <summary>
    /// Number of joint points
    /// </summary>
    private const int JointNum = 24;

    /// <summary>
    /// input image size
    /// </summary>
    public int InputImageSize;

    /// <summary>
    /// input image size (half)
    /// </summary>
    private float InputImageSizeHalf;

    /// <summary>
    /// column number of heatmap
    /// </summary>
    public int HeatMapCol;
    private float InputImageSizeF;
    private float lastCapturedTime;

    /// <summary>
    /// Column number of heatmap in 2D image
    /// </summary>
    private int HeatMapCol_Squared;
    
    /// <summary>
    /// Column nuber of heatmap in 3D model
    /// </summary>
    private int HeatMapCol_Cube;
    private float ImageScale;

    /// <summary>
    /// Buffer memory has 2D heat map
    /// </summary>
    private float[] heatMap2D;

    /// <summary>
    /// Buffer memory has offset 2D
    /// </summary>
    private float[] offset2D;
    
    /// <summary>
    /// Buffer memory has 3D heat map
    /// </summary>
    private float[] heatMap3D;
    
    /// <summary>
    /// Buffer memory hash 3D offset
    /// </summary>
    private float[] offset3D;
    private float unit;
    
    /// <summary>
    /// Number of joints in 2D image
    /// </summary>
    private int JointNum_Squared = JointNum * 2;
    
    /// <summary>
    /// Number of joints in 3D model
    /// </summary>
    private int JointNum_Cube = JointNum * 3;

    /// <summary>
    /// HeatMapCol * JointNum
    /// </summary>
    private int HeatMapCol_JointNum;

    /// <summary>
    /// HeatMapCol * JointNum_Squared
    /// </summary>
    private int CubeOffsetLinear;

    /// <summary>
    /// HeatMapCol * JointNum_Cube
    /// </summary>
    private int CubeOffsetSquared;

    /// <summary>
    /// For Kalman filter parameter Q
    /// </summary>
    public float KalmanParamQ;

    /// <summary>
    /// For Kalman filter parameter R
    /// </summary>
    public float KalmanParamR;

    /// <summary>
    /// Lock to update VNectModel
    /// </summary>
    private bool Lock = true;

    /// <summary>
    /// Use low pass filter flag
    /// </summary>
    public bool UseLowPassFilter;

    /// <summary>
    /// For low pass filter
    /// </summary>
    public float LowPassParam;

    public Text Msg;
    public float WaitTimeModelLoad = 10f;
    private float Countdown = 0;
    public Texture2D InitImg;

    private void Start()
    {
        // Initialize 
        HeatMapCol_Squared = HeatMapCol * HeatMapCol;
        HeatMapCol_Cube = HeatMapCol * HeatMapCol * HeatMapCol;
        HeatMapCol_JointNum = HeatMapCol * JointNum;
        CubeOffsetLinear = HeatMapCol * JointNum_Cube;
        CubeOffsetSquared = HeatMapCol_Squared * JointNum_Cube;

        heatMap2D = new float[JointNum * HeatMapCol_Squared];
        offset2D = new float[JointNum * HeatMapCol_Squared * 2];
        heatMap3D = new float[JointNum * HeatMapCol_Cube];
        offset3D = new float[JointNum * HeatMapCol_Cube * 3];
        unit = 1f / (float)HeatMapCol;
        InputImageSizeF = InputImageSize;
        InputImageSizeHalf = InputImageSizeF / 2f;
        ImageScale = InputImageSize / (float)HeatMapCol;// 224f / (float)InputImageSize;

        // Disabel sleep
        Screen.sleepTimeout = SleepTimeout.NeverSleep;

        useCache = File.Exists(modelCacheFile());
        if (useCache)
        {
            Debug.Log("Using cache");
        }
        //else
        //{
        //    // Init model
        //}
        _model = ModelLoader.Load(NNModel, Verbose);
        _worker = WorkerFactory.CreateWorker(WorkerType, _model, Verbose);
        StartCoroutine("WaitLoad");

    }

    private void OnApplicationQuit()
    {
        if (!useCache)
        {
            using (StreamWriter outputFile = new StreamWriter(modelCacheFile()))
            {
                foreach (var jointPointCacheArr in cachedJointPoints)
                {
                    string[] csvLines = new string[jointPointCacheArr.Length];

                    for (var i = 0; i < jointPointCacheArr.Length; i++)
                    {
                        csvLines[i] = jointPointCacheArr[i].toCsvLine();
                    }
                    var line = string.Join("\t", csvLines);

                    outputFile.WriteLine(line);
                }
            }
            Debug.Log("Cache written to " + Directory.GetCurrentDirectory() + "//" + modelCacheFile());
        }
    }

    private void readFromCache()
    {
        string[] lines = File.ReadAllLines(modelCacheFile());
       
        foreach (string line in lines)
        {
            string[] csvLines = line.Split('\t');
            var jointPointCacheArr = new VNectModel.JointPointCache[csvLines.Length];
            for (var i = 0; i < csvLines.Length; i++)
            {
                jointPointCacheArr[i] = new VNectModel.JointPointCache(csvLines[i]);
            }
            cachedJointPoints.Add(jointPointCacheArr);
        }
        // moving average
        for (var i = 0; i < cachedJointPoints.Count; i++)
        {
            var cnt = 0f;
            var avgpos3D = new float[cachedJointPoints[0].Length, 3];
            for (var j = 0; j < cachedJointPoints.Count; j++)
            {
                if (cachedJointPoints[j][0].timeFrame - cachedJointPoints[i][0].timeFrame < 0.1 &&
                   cachedJointPoints[j][0].timeFrame - cachedJointPoints[i][0].timeFrame > -0.1)
                {
                    cnt+= 1;
                    for (var k = 0; k < cachedJointPoints[0].Length; k++)
                    {
                        avgpos3D[k, 0] += cachedJointPoints[j][k].pos3Dx;
                        avgpos3D[k, 1] += cachedJointPoints[j][k].pos3Dy;
                        avgpos3D[k, 2] += cachedJointPoints[j][k].pos3Dz;
                    }
                }
            }
            for (var k = 0; k < cachedJointPoints[0].Length; k++)
            {
                cachedJointPoints[i][k].pos3Dx = avgpos3D[k, 0] / cnt;
                cachedJointPoints[i][k].pos3Dy = avgpos3D[k, 1] / cnt;
                cachedJointPoints[i][k].pos3Dz = avgpos3D[k, 2] / cnt;
            }
        }
        Debug.Log("Cache loaded: " + cachedJointPoints.Count);
    }

    private string modelCacheFile()
    {
        return videoCapture.videoClipName() + ".cache";
    }

    private void Update()
    {
        //Debug.Log(Time.realtimeSinceStartup);
        if (!Lock)
        {
            if (useCache)
            {
                PredictPoseFromCache();
            } else
            {
                UpdateVNectModel();
            }
        }
    }

    private IEnumerator WaitLoad()
    {
        if (useCache)
        {
            jointPoints = VNectModel.Init();
            readFromCache();
            PredictPoseFromCache();
        }
        else
        {
            inputs[inputName_1] = new Tensor(InitImg);
            inputs[inputName_2] = new Tensor(InitImg);
            inputs[inputName_3] = new Tensor(InitImg);

            // Create input and Execute model
            yield return _worker.StartManualSchedule(inputs);

            // Get outputs
            for (var i = 2; i < _model.outputs.Count; i++)
            {
                b_outputs[i] = _worker.PeekOutput(_model.outputs[i]);
            }

            // Get data from outputs
            offset3D = b_outputs[2].data.Download(b_outputs[2].shape);
            heatMap3D = b_outputs[3].data.Download(b_outputs[3].shape);

            // Release outputs
            for (var i = 2; i < b_outputs.Length; i++)
            {
                b_outputs[i].Dispose();
            }

            // Init VNect model
            jointPoints = VNectModel.Init();

            PredictPose();
            yield return new WaitForSeconds(WaitTimeModelLoad);
        }

        // Init VideoCapture
        videoCapture.Init(InputImageSize, InputImageSize);
        Lock = false;
        Msg.gameObject.SetActive(false);
    }

    private const string inputName_1 = "input.1";
    private const string inputName_2 = "input.4";
    private const string inputName_3 = "input.7";
    /*
    private const string inputName_1 = "0";
    private const string inputName_2 = "1";
    private const string inputName_3 = "2";
    */

    private void UpdateVNectModel()
    {
        input = new Tensor(videoCapture.MainTexture);
        if (inputs[inputName_1] == null)
        {
            inputs[inputName_1] = input;
            inputs[inputName_2] = new Tensor(videoCapture.MainTexture);
            inputs[inputName_3] = new Tensor(videoCapture.MainTexture);
            lastCapturedTime = videoCapture.getTimeElapsed();
        }
        else if (videoCapture.getTimeElapsed() - lastCapturedTime > 0.05)
        {
            inputs[inputName_3].Dispose();

            inputs[inputName_3] = inputs[inputName_2];
            inputs[inputName_2] = inputs[inputName_1];
            inputs[inputName_1] = input;
            lastCapturedTime = videoCapture.getTimeElapsed();
        } else
        {
            inputs[inputName_1] = input;
        }

        StartCoroutine(ExecuteModelAsync());
    }

    /// <summary>
    /// Tensor has input image
    /// </summary>
    /// <returns></returns>
    Tensor input = new Tensor();
    Dictionary<string, Tensor> inputs = new Dictionary<string, Tensor>() { { inputName_1, null }, { inputName_2, null }, { inputName_3, null }, };
    Tensor[] b_outputs = new Tensor[4];

    private IEnumerator ExecuteModelAsync()
    {
        // Create input and Execute model
        yield return _worker.StartManualSchedule(inputs);

        // Get outputs
        for (var i = 2; i < _model.outputs.Count; i++)
        {
            b_outputs[i] = _worker.PeekOutput(_model.outputs[i]);
        }

        // Get data from outputs
        offset3D = b_outputs[2].data.Download(b_outputs[2].shape);
        heatMap3D = b_outputs[3].data.Download(b_outputs[3].shape);
        
        // Release outputs
        for (var i = 2; i < b_outputs.Length; i++)
        {
            b_outputs[i].Dispose();
        }

        PredictPose();
    }

    private void PredictPoseFromCache()
    {
        while (cacheCursor < cachedJointPoints.Count &&
            cachedJointPoints[cacheCursor][0].timeFrame < videoCapture.getTimeElapsed())
        {
            cacheCursor++;
        }
        for (var i = 0; i < jointPoints.Length; i++)
        {
            cachedJointPoints[cacheCursor][i].updateJointPoint(jointPoints[i]);
        }
    }

    /// <summary>
    /// Predict positions of each of joints based on network
    /// </summary>
    private void PredictPose()
    {
        for (var j = 0; j < JointNum; j++)
        {
            var maxXIndex = 0;
            var maxYIndex = 0;
            var maxZIndex = 0;
            jointPoints[j].score3D = 0.0f;
            var jj = j * HeatMapCol;
            for (var z = 0; z < HeatMapCol; z++)
            {
                var zz = jj + z;
                for (var y = 0; y < HeatMapCol; y++)
                {
                    var yy = y * HeatMapCol_Squared * JointNum + zz;
                    for (var x = 0; x < HeatMapCol; x++)
                    {
                        float v = heatMap3D[yy + x * HeatMapCol_JointNum];
                        if (v > jointPoints[j].score3D)
                        {
                            jointPoints[j].score3D = v;
                            maxXIndex = x;
                            maxYIndex = y;
                            maxZIndex = z;
                        }
                    }
                }
            }
           
            jointPoints[j].Now3D.x = (offset3D[maxYIndex * CubeOffsetSquared + maxXIndex * CubeOffsetLinear + j * HeatMapCol + maxZIndex] + 0.5f + (float)maxXIndex) * ImageScale - InputImageSizeHalf;
            jointPoints[j].Now3D.y = InputImageSizeHalf - (offset3D[maxYIndex * CubeOffsetSquared + maxXIndex * CubeOffsetLinear + (j + JointNum) * HeatMapCol + maxZIndex] + 0.5f + (float)maxYIndex) * ImageScale;
            jointPoints[j].Now3D.z = (offset3D[maxYIndex * CubeOffsetSquared + maxXIndex * CubeOffsetLinear + (j + JointNum_Squared) * HeatMapCol + maxZIndex] + 0.5f + (float)(maxZIndex - 14)) * ImageScale;
        }

        // Calculate hip location
        var lc = (jointPoints[PositionIndex.rThighBend.Int()].Now3D + jointPoints[PositionIndex.lThighBend.Int()].Now3D) / 2f;
        jointPoints[PositionIndex.hip.Int()].Now3D = (jointPoints[PositionIndex.abdomenUpper.Int()].Now3D + lc) / 2f;

        // Calculate neck location
        jointPoints[PositionIndex.neck.Int()].Now3D = (jointPoints[PositionIndex.rShldrBend.Int()].Now3D + jointPoints[PositionIndex.lShldrBend.Int()].Now3D) / 2f;

        // Calculate head location
        var cEar = (jointPoints[PositionIndex.rEar.Int()].Now3D + jointPoints[PositionIndex.lEar.Int()].Now3D) / 2f;
        var hv = cEar - jointPoints[PositionIndex.neck.Int()].Now3D;
        var nhv = Vector3.Normalize(hv);
        var nv = jointPoints[PositionIndex.Nose.Int()].Now3D - jointPoints[PositionIndex.neck.Int()].Now3D;
        jointPoints[PositionIndex.head.Int()].Now3D = jointPoints[PositionIndex.neck.Int()].Now3D + nhv * Vector3.Dot(nhv, nv);

        // Calculate spine location
        jointPoints[PositionIndex.spine.Int()].Now3D = jointPoints[PositionIndex.abdomenUpper.Int()].Now3D;

        // Kalman filter
        foreach (var jp in jointPoints)
        {
            KalmanUpdate(jp);
        }

        // Low pass filter
        if (UseLowPassFilter)
        {
            foreach (var jp in jointPoints)
            {
                jp.PrevPos3D[0] = jp.Pos3D;
                for (var i = 1; i < jp.PrevPos3D.Length; i++)
                {
                    jp.PrevPos3D[i] = jp.PrevPos3D[i] * LowPassParam + jp.PrevPos3D[i - 1] * (1f - LowPassParam);
                }
                jp.Pos3D = jp.PrevPos3D[jp.PrevPos3D.Length - 1];
            }
        }

        var jointPointsCache = new VNectModel.JointPointCache[jointPoints.Length];
        for (var i = 0; i < jointPoints.Length; i++)
        {
            jointPointsCache[i] = new VNectModel.JointPointCache(jointPoints[i], videoCapture.getTimeElapsed());
        }
        cachedJointPoints.Add(jointPointsCache);
    }

    /// <summary>
    /// Kalman filter
    /// </summary>
    /// <param name="measurement">joint points</param>
    void KalmanUpdate(VNectModel.JointPoint measurement)
    {
        measurementUpdate(measurement);
        measurement.Pos3D.x = measurement.X.x + (measurement.Now3D.x - measurement.X.x) * measurement.K.x;
        measurement.Pos3D.y = measurement.X.y + (measurement.Now3D.y - measurement.X.y) * measurement.K.y;
        measurement.Pos3D.z = measurement.X.z + (measurement.Now3D.z - measurement.X.z) * measurement.K.z;
        measurement.X = measurement.Pos3D;
    }

	void measurementUpdate(VNectModel.JointPoint measurement)
    {
        measurement.K.x = (measurement.P.x + KalmanParamQ) / (measurement.P.x + KalmanParamQ + KalmanParamR);
        measurement.K.y = (measurement.P.y + KalmanParamQ) / (measurement.P.y + KalmanParamQ + KalmanParamR);
        measurement.K.z = (measurement.P.z + KalmanParamQ) / (measurement.P.z + KalmanParamQ + KalmanParamR);
        measurement.P.x = KalmanParamR * (measurement.P.x + KalmanParamQ) / (KalmanParamR + measurement.P.x + KalmanParamQ);
        measurement.P.y = KalmanParamR * (measurement.P.y + KalmanParamQ) / (KalmanParamR + measurement.P.y + KalmanParamQ);
        measurement.P.z = KalmanParamR * (measurement.P.z + KalmanParamQ) / (KalmanParamR + measurement.P.z + KalmanParamQ);
    }
}
