using System.Collections.Generic;
using UnityEngine;
using Unity.Sentis;
using TMPro;

public class SignLanguageManager : MonoBehaviour
{
    [Header("Sentis AI")]
    public ModelAsset fileModelAI;
    private Worker mesinAI;

    [Header("Kamus Isyarat (Urutan Dataset)")]
    public string[] vocabulary;

    [Header("AR Filter UI")]
    public RectTransform canvasRect;
    public RectTransform floatingTextContainer;
    public TextMeshProUGUI resultText;
    public float heightOffset = 300f;
    public float thresholdKeyakinan = 0.6f;
    public GameObject objekSolution;
    public float ukuranFontTerjemahan = 80f; 

    [Header("Optimasi Performa (Anti Patah-Patah)")]
    [Tooltip("Jalankan AI tiap X frame sekali. Makin besar angkanya, HP makin enteng.")]
    public int intervalInferenceFrame = 10; 

    private List<float[]> inputBuffer = new List<float[]>();
    private Vector2 currentHeadPos;
    private float hideTimer = 0f;
    private bool aiSiapDanValid = false;
    private int counterFrameSkor = 0; 

    void Awake()
    {
#if UNITY_ANDROID
        if (!UnityEngine.Android.Permission.HasUserAuthorizedPermission(UnityEngine.Android.Permission.Camera))
        {
            UnityEngine.Android.Permission.RequestUserPermission(UnityEngine.Android.Permission.Camera);
            StartCoroutine(TungguIzinKamera());
        }
        else
        {
            StartCoroutine(NyalakanMediaPipe());
        }
#else        
        StartCoroutine(NyalakanMediaPipe());
#endif
    }

    void Start()
    {
        if (fileModelAI == null)
        {
            Debug.LogError("[SignLingo ERROR] File Model AI BELUM dimasukkan ke Inspector!");
            aiSiapDanValid = false;
            return;
        }

        try
        {
            var runtimeModel = ModelLoader.Load(fileModelAI);
            mesinAI = new Worker(runtimeModel, BackendType.CPU);
            aiSiapDanValid = true;

            
            if (resultText != null)
            {
                resultText.fontSize = ukuranFontTerjemahan;
                resultText.alignment = TextAlignmentOptions.Center;
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"[SignLingo ERROR] Gagal memuat Model AI: {e.Message}");
            aiSiapDanValid = false;
        }

        if (floatingTextContainer != null) floatingTextContainer.gameObject.SetActive(false);
    }

    System.Collections.IEnumerator TungguIzinKamera()
    {
        while (!UnityEngine.Android.Permission.HasUserAuthorizedPermission(UnityEngine.Android.Permission.Camera))
        {
            yield return new WaitForSeconds(0.5f);
        }
        yield return NyalakanMediaPipe();
    }

    System.Collections.IEnumerator NyalakanMediaPipe()
    {
        yield return new WaitForSeconds(0.6f);
        if (objekSolution != null)
        {
            objekSolution.SetActive(true);
        }
    }

    void Update()
    {
        if (floatingTextContainer != null && floatingTextContainer.gameObject.activeSelf)
        {
            float xPos = (0.5f - currentHeadPos.x) * canvasRect.rect.width;
            float yPos = ((1f - currentHeadPos.y) - 0.5f) * canvasRect.rect.height;
            floatingTextContainer.anchoredPosition = new Vector2(xPos, yPos + heightOffset);
        }

        if (hideTimer > 0)
        {
            hideTimer -= Time.deltaTime;
            if (hideTimer <= 0)
            {
                floatingTextContainer.gameObject.SetActive(false);
            }
        }
    }

    public void SusunData258(List<Vector4> pose, List<Vector3> leftHand, List<Vector3> rightHand)
    {
        if (!aiSiapDanValid) return;

        List<float> frameData = new List<float>();

        if (pose != null && pose.Count == 33)
        {
            currentHeadPos = new Vector2(pose[0].x, pose[0].y);
            foreach (var p in pose)
            {
                frameData.Add(p.x); frameData.Add(p.y); frameData.Add(p.z); frameData.Add(p.w);
            }
        }
        else
        {
            frameData.AddRange(new float[132]);
        }

        if (leftHand != null && leftHand.Count == 21)
        {
            foreach (var h in leftHand)
            {
                frameData.Add(h.x); frameData.Add(h.y); frameData.Add(h.z);
            }
        }
        else
        {
            frameData.AddRange(new float[63]);
        }

        if (rightHand != null && rightHand.Count == 21)
        {
            foreach (var h in rightHand)
            {
                frameData.Add(h.x); frameData.Add(h.y); frameData.Add(h.z);
            }
        }
        else
        {
            frameData.AddRange(new float[63]);
        }

        inputBuffer.Add(frameData.ToArray());

        
        if (inputBuffer.Count > 60)
        {
            inputBuffer.RemoveAt(0);
        }

        
        if (inputBuffer.Count == 60)
        {
            counterFrameSkor++;
            if (counterFrameSkor >= intervalInferenceFrame)
            {
                ProsesKlasifikasi();
                counterFrameSkor = 0; 
            }
        }
    }

    void ProsesKlasifikasi()
    {
        if (mesinAI == null) return;

        float[] flatData = new float[60 * 258];
        for (int i = 0; i < 60; i++) System.Array.Copy(inputBuffer[i], 0, flatData, i * 258, 258);

        try
        {
            Tensor<float> inputTensor = new Tensor<float>(new TensorShape(1, 60, 258), flatData);
            mesinAI.Schedule(inputTensor);

            var outputRaw = mesinAI.PeekOutput();
            if (outputRaw == null)
            {
                inputTensor.Dispose();
                return;
            }

            var outputTensor = outputRaw as Tensor<float>;
            if (outputTensor == null)
            {
                inputTensor.Dispose();
                return;
            }

            int predictedIndex = GetArgMax(outputTensor);
            ShowResult(predictedIndex);

            inputTensor.Dispose();
        }
        catch (System.Exception e)
        {
            Debug.LogError($"[SignLingo Performance Guard] Error: {e.Message}");
        }
    }

    int GetArgMax(Tensor<float> tensor)
    {
        if (tensor == null) return -1;

        float[] probabilities = tensor.DownloadToArray();
        float maxVal = -1f;
        int maxIdx = -1;

        for (int i = 0; i < probabilities.Length; i++)
        {
            if (probabilities[i] > maxVal)
            {
                maxVal = probabilities[i];
                maxIdx = i;
            }
        }

        if (maxVal < thresholdKeyakinan) return -1;

        return maxIdx;
    }

    void ShowResult(int index)
    {
        if (index >= 0 && index < vocabulary.Length)
        {
            resultText.text = vocabulary[index];
            floatingTextContainer.gameObject.SetActive(true);
            hideTimer = 1.8f; 
        }
    }

    void OnDestroy()
    {
        mesinAI?.Dispose();
    }
}

