module hdtwo

open FSharp.Data
open FSharp.Charting
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.Statistics

let csv_path = @"C:\home\myproj\study_ml\abnormaly_detection\data\car2.csv"
let output_path = @"C:\home\myproj\study_ml\abnormaly_detection\fsharp\Hotelling\output\"
type CarDB = CsvProvider<"C:/home/myproj/study_ml/abnormaly_detection/data/car2.csv">

let plotAbnormality (xs :List<float>, th:float)  = 
    let mutable chart_list = []
    let n = Seq.length xs

    let chart1 =  Chart.Point [for i in 0 .. n - 1 -> (i + 1,  xs.[i]) ]
    let output_file = output_path + @"abnomality.png" 

    let chart2 = Chart.Line [ for x in 0.0 .. 200.0 -> (x, th) ]

    let chart_comb = Chart.Combine [chart1; chart2]

    Chart.Save output_file chart_comb
    
    printf "%s\n"  output_file
    None

let getData = 
    let car = CarDB.Load(csv_path)
        
    let weight = [ for row in car.Rows -> (float row.Weight)]
    let height = [ for row in car.Rows -> (float row.Height)]
    //weight, height
    let n = List.length weight

    //let data = DenseMatrix.init 2 n (fun i j -> if i == 0 then weight.Item j else height.Item j)
    let arr = array2D [ weight; height ]
    let data = Matrix.Build.DenseOfArray  arr

    data.Transpose()

let calcMahalanobis(x:Vector<float>, mu_hat:Vector<float>, sig_hat_inv:Matrix<float>) =
    let x_minus_mu_hat = x - mu_hat
    let a = x_minus_mu_hat * sig_hat_inv * x_minus_mu_hat

    a
    


let calcAbnormality (x:Matrix<float>) =
    let n = float x.RowCount
    let weights = x.Column(0)
    let heights = x.Column(1)

    // \hat μ
    let weight_mean = Statistics.Mean weights
    let height_mean = Statistics.Mean heights
    let mu_arr = [|weight_mean; height_mean|]
    let mu_hat = Vector.Build.DenseOfArray mu_arr

    // \hat Σ
    let sig_arr = array2D [ weights - weight_mean; heights - height_mean ]
    let x_minus_mu = Matrix.Build.DenseOfArray  sig_arr
    let sig_hat = (x_minus_mu * x_minus_mu.Transpose()) / n
    let sig_hat_inv = sig_hat.Inverse()

    // 異常度の計算
    let mutable abnormality = []
    for d in  x.EnumerateRows() do
        let a = calcMahalanobis(d, mu_hat, sig_hat_inv)
        abnormality <- a :: abnormality

    List.rev abnormality

