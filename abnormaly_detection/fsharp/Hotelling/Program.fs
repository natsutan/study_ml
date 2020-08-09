// Learn more about F# at https://fsharp.org
// See the 'F# Tutorial' project for more help.
open MathNet.Numerics.Distributions

let hd1d = false



[<EntryPoint>]
let main argv =
    if hd1d then
        let weights = hdoned.getWeights
        hdoned.plotWeights weights |> ignore

        let abnormality = hdoned.calcAbnormality weights
        let threshhold =  ChiSquared.InvCDF(1.0, 0.99)
        printfn "閾値 = %A" threshhold

        hdoned.plotAbnormality (abnormality, threshhold) |> ignore
    
    let data = hdtwo.getData
    let abnormality = hdtwo.calcAbnormality data
    
    let threshhold = ChiSquared.InvCDF(1.0, 0.99)
    hdtwo.plotAbnormality (abnormality, threshhold) |> ignore
    

    0 // return an integer exit code
