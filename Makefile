.PHONY:run
run:
	cd build; ./FileReader&
	cd build; ./Fitter&
#	cd build; ./FileWriter&

.PHONY: stop
stop:
	pkill FileReader || true
	pkill Fitter || true
	pkill FileWriter || true
	
.PHONY: start_signaling
start_signaling:
	./build/zmq_signaling&
	
.PHONY: end_signaling
end_signaling:
	pkill zmq_signaling

