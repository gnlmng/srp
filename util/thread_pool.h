#pragma once

#include <functional>
#include <memory>
#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <atomic>

class Thread_pool {
	
	using ulock = std::unique_lock <std::mutex>;
	
public:
	
	Thread_pool() :
	m_done(false),
	m_stop(false),
	m_cnt_waiting(0)
	{
		resize(std::max(1u, std::thread::hardware_concurrency()));
	}
	
	Thread_pool(int32_t _size) :
	m_done(false),
	m_stop(false),
	m_cnt_waiting(0)
	{
		resize(_size);
	}
	
	~Thread_pool() {
		stop(true);
	}
	
	inline int32_t size() const {
		return m_threads.size();
	}
	
	inline int32_t idle() const {
		return m_cnt_waiting;
	}
	
	inline std::thread& operator [] (const int32_t index) {
		return *m_threads[index];
	}
	
	void resize(int32_t new_size) {
		if (m_stop || m_done) {
			return;
		}
		int32_t prev_size = size();
		if (new_size == prev_size) {
			return;
		}
		if (new_size > prev_size) {
			m_threads.resize(new_size);
			m_flags.resize(new_size);
			for (int32_t i = prev_size; i < new_size; i++) {
				m_flags[i] = std::make_shared <std::atomic <bool>> (false);
				m_inf_func(i);
			}
			return;
		}
		for (int32_t i = prev_size - 1; i >= 0; i--) {
			*m_flags[i] = true;
			m_threads[i]->detach();
		}
		{
			ulock lock(m_mutex);
			m_condition_variable.notify_all();
		}
		m_threads.resize(new_size);
		m_flags.resize(new_size);
	}
	
	inline void clear() {
		std::function <void (int32_t)>* f;
		while (m_queue.pop(f)) {
			delete(f);
		}
	}
	
	inline std::function <void (int32_t)> pop() {
		std::function <void (int32_t)>* f = nullptr;
		m_queue.pop(f);
		std::unique_ptr <std::function <void (int32_t)>> function(f);
		std::function <void (int32_t)> bad;
		return f ? *f : bad;
	}
	
	void stop(bool wait = false) {
		if (wait) {
			if (m_done || m_stop) {
				return;
			}
			m_done = true;
		} else {
			if (m_stop) {
				return;
			}
			m_stop = true;
			for (int32_t i = 0; i < size(); i++) {
				*m_flags[i] = true;
			}
			clear();
		}
		{
			ulock lock(m_mutex);
			m_condition_variable.notify_all();
		}
		for (int32_t i = 0; i < size(); i++) {
			if (m_threads[i]->joinable()) {
				m_threads[i]->join();
			}
		}
		clear();
		m_threads.clear();
		m_flags.clear();
	}
	
	template <typename F, typename... A>
	auto push(F&& function, A&&... args) -> std::future <decltype (function (0, args...))> {
		auto p = std::make_shared <std::packaged_task <decltype (function (0, args...)) (int32_t)>>(
		std::bind(std::forward <F> (function), std::placeholders::_1, std::forward <A> (args)...));
		auto f = new std::function <void (int32_t)> ([p] (int32_t id) { (*p) (id); });
		m_queue.push(f);
		ulock lock(m_mutex);
		m_condition_variable.notify_one();
		return p->get_future();
	}
	
	template <typename F> auto push(F&& function) -> std::future <decltype (function (0))> {
		auto p = std::make_shared <std::packaged_task <decltype (function (0)) (int32_t)>> (std::forward <F> (function));
		auto f = new std::function <void (int32_t)> ([p] (int32_t id) { (*p) (id); });
		m_queue.push(f);
		ulock lock(m_mutex);
		m_condition_variable.notify_one();
		return p->get_future();
	}
	
	void wait_all() {
		while (m_cnt_waiting < (int32_t) m_threads.size());
		ulock lock(m_wait_all_mutex);
		m_wait_all_condition_variable.wait(lock, [this] () {
			return this->m_queue.empty() && this->m_cnt_waiting == (int32_t) this->m_threads.size();
		});
	}
	
private:
	
	template <typename A> struct Job_queue {
		
		inline void push(const A& value) {
			ulock lock(this->m_mutex);
			this->m_queue.push(value);
		}
		
		inline bool pop(A& value) {
			ulock lock(this->m_mutex);
			if (this->m_queue.empty()) {
				return false;
			}
			value = this->m_queue.front();
			this->m_queue.pop();
			return true;
		}
		
		inline bool empty() {
			ulock lock(this->m_mutex);
			return this->m_queue.empty();
		}
		
	private:
		
		std::queue <A> m_queue;
		std::mutex m_mutex;
		
	};
	
private:
	
	Thread_pool(const Thread_pool&);
	Thread_pool(Thread_pool&&);
	Thread_pool& operator = (const Thread_pool&);
	Thread_pool& operator = (Thread_pool&&);
	
	std::vector <std::unique_ptr <std::thread>> m_threads;
	std::vector <std::shared_ptr <std::atomic <bool>>> m_flags;
	Job_queue <std::function <void (int32_t)>*> m_queue;
	std::atomic <bool> m_done, m_stop;
	std::atomic <int32_t> m_cnt_waiting;
	std::mutex m_mutex;
	std::condition_variable m_condition_variable;
	
	std::mutex m_wait_all_mutex;
	std::condition_variable m_wait_all_condition_variable;
	
private:
	
	void m_inf_func(int32_t index) {
		std::shared_ptr <std::atomic <bool>> flag(m_flags[index]);
		auto inf_f = [this, index, flag] () -> void {
			std::atomic <bool>& ref_flag = *flag;
			std::function <void (int32_t)>* f;
			bool is_pop = m_queue.pop(f);
			while (true) {
				m_wait_all_condition_variable.notify_all();
				while (is_pop) {
					std::unique_ptr <std::function <void (int32_t)>> ff(f);
					(*f) (index);
					if (ref_flag) {
						return;
					}
					is_pop = m_queue.pop(f);
				}
				ulock lock(m_mutex);
				m_cnt_waiting++;
				m_condition_variable.wait(lock, [this, &f, &is_pop, &ref_flag] () {
					m_wait_all_condition_variable.notify_all();
					is_pop = m_queue.pop(f);
					return is_pop || m_done || ref_flag;
				});
				m_cnt_waiting--;
				if (!is_pop) {
					return;
				}
			}
		};
		m_threads[index].reset(new std::thread (inf_f));
	}
	
};
