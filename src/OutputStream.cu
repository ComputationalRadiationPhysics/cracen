#include "OutputStream.h"

/* defines how the output is written into the output file */
std::ostream& operator<<(std::ostream& lhs, const Output& rhs) {
	lhs << "{startValue=" << rhs.startValue << ",endValue=" << rhs.endValue << ",extremumPos=" << rhs.extremumPos  << ",extremumValue=" << rhs.extremumValue << "}";
	return lhs;
}

void OutputStream::run() {
	while(!done) {
		Output* o = oBuffer.reserveTail();
		file << (*o) << std::endl;
		oBuffer.freeTail();
	}
	/* oBuffer komplett leer machen*/
	while(!oBuffer.isEmpty()) {
		Output* o = oBuffer.reserveTail();
		file << (*o) << std::endl;
		oBuffer.freeTail();
	}
	/* Close Filestream */
	file.close();
}

OutputStream::OutputStream(const std::string& file) :
	file(file.c_str(), std::ofstream::out),
	oBuffer(CHUNK_BUFFER_COUNT),
	done(false)
{
	run_helper(this);
}

Ringbuffer<Output>* OutputStream::getBuffer() {
	return &oBuffer;
}
void OutputStream::finish() {
	done = true;
}
