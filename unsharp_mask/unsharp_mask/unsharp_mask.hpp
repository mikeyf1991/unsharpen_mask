#ifndef _UNSHARP_MASK_HPP_
#define _UNSHARP_MASK_HPP_

#define BDIM 32.0f

#include "blur.hpp"
#include "add_weighted.hpp"
#include "ppm.hpp"
#include "common.h"

void d_unsharp_mask(unsigned char *out, const unsigned char *in,
	const int blur_radius,
	const unsigned w, const unsigned h, const unsigned nchannels)

{
	std::vector<unsigned char> blur1, blur2, blur3;

	float ThreadsX	= ceil(w / BDIM); float ThreadsY = ceil(h / BDIM);

	int maxThreadsX, maxThreadsY;
	cudaDeviceGetAttribute(&maxThreadsX, cudaDevAttrMaxGridDimX, 0);
	cudaDeviceGetAttribute(&maxThreadsY, cudaDevAttrMaxGridDimY, 0);

	dim3 blocksize(BDIM, BDIM);

	if (ThreadsX > maxThreadsX)
		ThreadsX = maxThreadsX;
	if (ThreadsY > maxThreadsY)
		ThreadsY = maxThreadsY;

	dim3 gridSize(ThreadsX, ThreadsY);

	size_t imgSize = w * h * nchannels;

	blur1.resize(imgSize);
	blur2.resize(imgSize);
	blur3.resize(imgSize);

	unsigned char *d_Blur1, *d_Blur2, *d_Blur3, *d_image_in, *d_image_out;

	cudaMalloc(&d_Blur1, imgSize);
	cudaMalloc(&d_Blur2, imgSize);
	cudaMalloc(&d_Blur3, imgSize);
	cudaMalloc(&d_image_in, imgSize);
	cudaMalloc(&d_image_out, imgSize);


	cudaMemcpy(d_image_in, in, imgSize, cudaMemcpyHostToDevice);

	d_blur <<<gridSize, blocksize>>>(d_Blur1, d_image_in, blur_radius, w, h, nchannels);
	d_blur <<<gridSize, blocksize>>>(d_Blur2, d_Blur1, blur_radius, w, h, nchannels);
	d_blur <<<gridSize, blocksize>>>(d_Blur3, d_Blur2, blur_radius, w, h, nchannels);
	d_add_weighted<<<gridSize, blocksize>>>(d_image_out, d_image_in, 1.5f, d_Blur3, -0.5f, 0.0f, w, h, nchannels);

	cudaMemcpy(out, d_image_out, imgSize, cudaMemcpyDeviceToHost);

	cudaFree(&d_Blur1);
	cudaFree(&d_Blur2);
	cudaFree(&d_Blur3);
	cudaFree(&d_image_in);
	cudaFree(&d_image_out);
}

void unsharp_mask(unsigned char *out, const unsigned char *in,
	const int blur_radius,
	const unsigned w, const unsigned h, const unsigned nchannels)
{
	std::vector<unsigned char> blur1, blur2, blur3;

	blur1.resize(w * h * nchannels);
	blur2.resize(w * h * nchannels);
	blur3.resize(w * h * nchannels);

	blur(blur1.data(), in, blur_radius, w, h, nchannels);
	blur(blur2.data(), blur1.data(), blur_radius, w, h, nchannels);
	blur(blur3.data(), blur2.data(), blur_radius, w, h, nchannels);
	add_weighted(out, in, 1.5f, blur3.data(), -0.5f, 0.0f, w, h, nchannels);
}

#endif // _UNSHARP_MASK_HPP_
