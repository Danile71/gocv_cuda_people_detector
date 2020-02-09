package main

import (
	"fmt"
	"image/color"

	"gocv.io/x/gocv"
	"gocv.io/x/gocv/cuda"
)

func main() {
	deviceID := "0"

	webcam, err := gocv.OpenVideoCapture(deviceID)
	if err != nil {
		fmt.Printf("Error opening video capture device: %v\n", deviceID)
		return
	}
	defer webcam.Close()

	window := gocv.NewWindow("Capture Window")
	defer window.Close()

	img := gocv.NewMat()
	defer img.Close()

	hog := cuda.CreateHOG()

	hog.SetSVMDetector(hog.GetDefaultPeopleDetector())

	gpumat := cuda.NewGpuMat()
	defer gpumat.Close()

	graygpumat := cuda.NewGpuMat()
	defer graygpumat.Close()

	fmt.Printf("Start reading device: %v\n", deviceID)
	for {
		if ok := webcam.Read(&img); !ok {
			fmt.Printf("Device closed: %v\n", deviceID)
			return
		}
		if img.Empty() {
			continue
		}
		gpumat.Upload(img)

		cuda.CvtColor(gpumat, &graygpumat, gocv.ColorBGRToGray)
		rects := hog.DetectMultiScale(graygpumat)
		for _, rect := range rects {
			gocv.Rectangle(&img, rect, color.RGBA{0, 0, 255, 0}, 2)
		}

		window.IMShow(img)
		if window.WaitKey(1) == 27 {
			break
		}
	}
}
