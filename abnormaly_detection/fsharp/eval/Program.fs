// Learn more about F# at https://fsharp.org
// See the 'F# Tutorial' project for more help.
open FSharp.Charting

type Sample = {score:float; anomality:bool}
let output_path = @"C:\home\myproj\study_ml\abnormaly_detection\fsharp\eval\output\"

let makeData :List<Sample> = 
//    let mutable data = []
    let scores = [0.19; 0.86; 0.17; 0.12; 0.04; 0.78; 0.16; 0.51; 0.57; 0.27]
    let anomalies = [false; true;false;false;false;true;false;true;false;false]

    List.map2 (fun s a -> {score=s; anomality = a}) scores anomalies

let calcRecallDetection(data:List<Sample>) = 
    let N = List.length data
    let recall:float[] = Array.create N 0.0
    let detection:float[] = Array.create N 0.0

    let anom_cnt = List.filter (fun x -> x.anomality) data |> List.length
    let norm_cnt = N - anom_cnt

    for i in 0 .. 9 do
        let d = data.[0..i]
        let danom = List.filter (fun x -> x.anomality) d |> List.length
        let dnorm = norm_cnt - (i+1 - danom)

        Array.set recall i (float(danom) / float(anom_cnt))
        Array.set detection  i (float(dnorm) / float(norm_cnt))


    (recall,detection)


let plot(data:List<Sample>, recall:float[], detection:float[]) = 

    let mutable recall_plot = [(1.0, 0.0)]
    let mutable detection_plot = [(1.0, 1.0)]
    let n = List.length data
    
    for i in 0 .. n - 1 do
        let score = data.[i].score
        recall_plot <- (score, recall.[i]) :: recall_plot
        detection_plot <- (score, detection.[i]) :: detection_plot
            
    let chart1 = Chart.Line(recall_plot)
    let chart2 = Chart.Line(detection_plot)
    
    let chart_comb = Chart.Combine([chart1; chart2]).WithYAxis(Max = 1.0, Min = 0.0)
    let output_file = output_path + @"fvalue.png" 
    Chart.Save output_file chart_comb

    printf "%s\n"  output_file



[<EntryPoint>]
let main argv =
    let data = makeData
    let sorted_data = List.sortBy (fun elem -> elem.score) data |> List.rev

    let recall, detection = calcRecallDetection sorted_data
    
    plot (sorted_data, recall, detection) |> ignore

    printfn "%A" recall
    printfn "%A" detection

    0 // return an integer exit code
