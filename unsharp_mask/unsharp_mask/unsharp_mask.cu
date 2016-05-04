#include "unsharp_mask.hpp"
#include "common.h"

// Apply an unsharp mask to the 24-bit PPM loaded from the file path of
// the first input argument; then write the sharpened output to the file path
// of the second argument. The third argument provides the blur radius.


void writeToFile(std::ofstream file, float CPU_Time, int i, float GPU_Time)
{
	file << CPU_Time << "," << i << "," << GPU_Time << "\n";
}


int main(int argc, char *argv[])
{
	std::cout << "opening image \n";
	const char *ifilename = argc > 1 ? argv[1] : "images/ghost-town-8k.ppm";
	const char *ofilename = argc > 2 ? argv[2] : "images/out.ppm";
	const int blur_radius = argc > 3 ? std::atoi(argv[3]) : 5;
	const char *ofilename1 = "images/GPU Processed.ppm";

	std::ofstream file;

	int blockSize, gridSize;

	ppm img;
	std::vector<unsigned char> data_in, data_sharp;

	img.read(ifilename, data_in);
	data_sharp.resize(img.w * img.h * img.nchannels);

	file.open("Test.csv");
	file << "CPU TEST" << ',' << ',' << "GPU TEST" << '\n';

	/*
	///////////////////////////////////////////////////
						CPU testing
	///////////////////////////////////////////////////
	*/

	for (int i = 0; i < 5; ++i)
	{
		std::cout << "CPU unsharp Test started \n";
		auto t1 = std::chrono::steady_clock::now();

		unsharp_mask(data_sharp.data(), data_in.data(), blur_radius,
			img.w, img.h, img.nchannels);

		auto t2 = std::chrono::steady_clock::now();
		auto timer = std::chrono::duration<double>(t2 - t1).count();
		std::cout << timer << " seconds.\n";

		file << timer << ',' << "run" + i << ',';

		std::cout << "unsharp finished \n";
		//Writes to file for checking results !!!!!!!!!
		std::cout << "creating finished image \n";
		//img.write(ofilename, data_sharp);
		//std::cout << "press enter";  std::cin.get();

		/*
		///////////////////////////////////////////////////
		GPU testing
		///////////////////////////////////////////////////
		*/

		std::cout << "GPU unsharp Test started \n";
		t1 = std::chrono::steady_clock::now();

		d_unsharp_mask(data_sharp.data(), data_in.data(), blur_radius,
			img.w, img.h, img.nchannels);

		t2 = std::chrono::steady_clock::now();
		timer = std::chrono::duration<double>(t2 - t1).count();
		std::cout << timer << " seconds.\n";

		file << timer << '\n';

		std::cout << "unsharp finished \n";
		//Writes to file for checking results !!!!!!!!!
		std::cout << "creating finished image \n";

		if (i==0)
			img.write(ofilename, data_sharp);

		if (i == 4)
			img.write(ofilename1, data_sharp);
	}
	std::cout << "press enter";  std::cin.get();

	return 0;
}

