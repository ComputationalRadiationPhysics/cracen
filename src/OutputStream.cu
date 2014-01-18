#include "OutputStream.h"

void OutputStream::run() {

}

OutputStream::OutputStream(std::string file) :
	file(file, std::ofstream::out),
	done(false)
{
	while(!done) {
		Output* o = oBuffer.reserveTail();
		file << o << std::endl;
		oBuffer.freeTail(o);
	}
	/* oBuffer komplett leer machen*/
	while(!oBuffer.isEmpty()) {
		Output* o = oBuffer.reserveTail();
		file << o << std::endl;
		oBuffer.freeTail(o);
	}
	/* Close Filestream */
	file.close();
}

void OutputStream::finish() {
	done = true;
}
