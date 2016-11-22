#pragma once

typedef unsigned int CracenHandle;

struct CracenEnum {
	enum Hook {
		PREPROCESSING,
		KERNEL,
		POSTPROCESSING
	};
	
	enum Role {
		Source,
		Intermediate,
		Sink
	};
};
extern "C" void CracenInit(CracenHandle* cracenHandle, CracenEnum::Role role);
extern "C" void CracenSetSendPolicy(CracenHandle* cracen, unsigned int SendPolicy);
extern "C" void CracenBind(CracenHandle* cracen, void* fn, CracenEnum::Hook hook);
extern "C" void CracenRelease(CracenHandle* cracen);
