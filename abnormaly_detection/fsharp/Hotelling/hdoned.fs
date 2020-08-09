module hdoned

open FSharp.Data
open FSharp.Charting

let csv_path = @"C:\home\myproj\study_ml\abnormaly_detection\data\car.csv"
let output_path = @"C:\home\myproj\study_ml\abnormaly_detection\fsharp\Hotelling\output\"
type CarDB = CsvProvider<"C:/home/myproj/study_ml/abnormaly_detection/data/car.csv">

let getWeights = 
    let car = CarDB.Load(csv_path)
    
    [ for row in car.Rows -> (float row.Weight)]
    
let plotWeights (xs :List<float>)  = 
    let n = Seq.length xs
    let chart =  Chart.Point [for i in 0 .. n - 1 -> (i + 1,  xs.[i]) ]
    let output_file = output_path + @"weights.png" 
    Chart.Save output_file chart
    
    printf "%s\n"  output_file
    None

let statWeight (xs:List<float>) =
    let mu = List.average xs
    let sigma2 = List.map (fun x -> (x - mu) ** 2.0) xs  |> List.average

    (mu, sigma2)

let calcAbnormality (xs:List<float>) =
    let mu, sigma2 = statWeight xs
    List.map (fun x -> (x - mu) ** 2.0 / sigma2) xs
   
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
