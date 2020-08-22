// Learn more about F# at https://fsharp.org
// See the 'F# Tutorial' project for more help.
open MathNet.Numerics.Distributions
open FSharp.Charting

type GaussDistri = { mu :float ; sigma :float}
let randgen = System.Random()
let output_path = @"C:\home\myproj\study_ml\abnormaly_detection\fsharp\abnormal_data\output\"


let sampleData(gd0:GaussDistri, gd1:GaussDistri, pi:float):float = 
    
    if randgen.NextDouble() < pi then
        Normal(mean=gd0.mu, stddev= gd0.sigma).Sample()
    else
        Normal(mean=gd1.mu, stddev= gd1.sigma).Sample()


let generateData = 
    let N = 1000
    let pi0 = 0.6
    let gd0:GaussDistri = {mu = 3.0; sigma =  0.5}
    let gd1:GaussDistri = {mu = 0.0; sigma =  3.0}

    [| for _ in 1 .. N -> sampleData(gd0, gd1, pi0) |]


let plotData (data:double[]) = 
    let n = Seq.length data

    let chart =  Chart.Point [for i in 0 .. n - 1 -> (i + 1,  data.[i]) ]
    let output_file = output_path + @"em.png" 

    Chart.Save output_file chart
    printf "%s\n"  output_file
    None

let calcMu(x:double[], q:double[], pi:double[]):float =
    let numerator = Array.map2 (fun x y -> x * y) x q |> Array.sum
    let denominator = Array.sum q

    numerator / denominator

let calcSigma(x:double[], q:double[], mu:float, pi:float) :float = 
    let x_minus_mu = Array.map (fun x -> (x - mu)**2.0 ) x
    let numerator = Array.map2 (fun x y -> x * y) q x_minus_mu|> Array.sum
    
    let denominator = Array.sum q

    sqrt (numerator / denominator)


let em (data:double[]) = 
    //初期値
    let mutable pi0 = 0.5
    let mutable pi1 = 0.5
    let mutable mu0 = 5.0
    let mutable mu1 = -5.0
    let mutable sigma0 = 1.0
    let mutable sigma1 = 5.0

    let mutable qn0 = 0.0
    let mutable qn1 = 0.0

    for _ in 0 ..10 do
        let piN0 = Array.map (fun x -> pi0 * Normal.PDF(mu0, sigma0, x)) data
        let piN1 = Array.map (fun x -> pi1 * Normal.PDF(mu1, sigma1, x)) data
        let qn0 = Array.map2 (fun p0 p1 -> p0 / (p0 + p1)) piN0 piN1
        let qn1 = Array.map2 (fun p0 p1 -> p1 / (p0 + p1)) piN0 piN1
        pi0 <- Array.average qn0
        pi1 <- Array.average qn1
        mu0 <- calcMu(data, qn0, piN0)
        mu1 <- calcMu(data, qn1, piN1)
        sigma0 <- calcSigma(data, qn0, mu0, pi0)
        sigma1 <- calcSigma(data, qn1, mu1, pi1)

    let gd0:GaussDistri = {mu = mu0; sigma =  sigma0}
    let gd1:GaussDistri = {mu = mu1; sigma =  sigma1}

    gd0, gd1, (pi0, pi1)
    

[<EntryPoint>]
let main argv =
    let data = generateData
    plotData data |> ignore
    let gd0, gd1, pis = em data

    printf "gd0 %A\n" gd0
    printf "gd1 %A\n" gd1
    printf "%A\n" pis


    0 // return an integer exit code
