#include "OutputStream.h"

/* defines how the output is written into the output file */
std::ostream& operator<<(std::ostream& lhs, const Output& rhs) {
	lhs << rhs.status << "	" << rhs.startValue << "	" << rhs.endValue << "	" << rhs.extremumPos  << "	" << rhs.extremumValue;
	return lhs;
}

void OutputStream::run() {
	file << "status	startValue	endValue	extremumPos	extremumValue" << std::endl;
	while(!oBuffer.isFinished()) {
		Output* o = oBuffer.reserveTail();
		if(o != NULL) {
			file << (*o) << std::endl;
			oBuffer.freeTail();
		}
	}
	/* Close Filestream */
	file.close();
}

OutputStream::OutputStream(const std::string& file, int producer) :
	file(file.c_str(), std::ofstream::out),
	oBuffer(CHUNK_BUFFER_COUNT, producer),
	done(false)
{
	pthread_create(&thread_tid, NULL, run_helper, this);
}

Ringbuffer<Output>* OutputStream::getBuffer() {
	return &oBuffer;
}

void OutputStream::join() {
	pthread_join(thread_tid, NULL);
}
